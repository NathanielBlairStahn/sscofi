import numpy as np
import pandas as pd

def get_contributions_df(members_df, membership_df, preferred_df, other_equity_df):
    """Combines data from all four data tables into one list of transactions.
    """
    #This version modifies the input dataframes:
    #membership_df['type'] = 'membership'
    #preferred_df['type'] = 'preferred'
    #other_equity_df['type'] = 'other'

    #Instead, use assign to make a copy of the dataframes rather than modifying them in place:
    membership_df = membership_df.assign(type='membership')
    preferred_df = preferred_df.assign(type='preferred')
    other_equity_df = other_equity_df.assign(type='other')

    equity_dfs = [membership_df, preferred_df, other_equity_df]
    contributions_df = pd.concat(equity_dfs, join='inner', ignore_index=True)

    contributions_df = members_df.merge(contributions_df, on='name')
    #contributions_df.rename(columns={'id': 'member_id'}, inplace=True)

    return contributions_df

def split_by_year(contributions_df, current_year, return_future=True):
    """Splits a list of transactions into 3 lists:
    one for the current year, one for past years, and one for future years.
    """
    year = pd.to_datetime(contributions_df['date']).apply(lambda date: date.year)
    past = contributions_df[year < current_year]
    current = contributions_df[year == current_year]
    if return_future:
        future = contributions_df[year > current_year]
        return past, current, future
    else:
        return past, current

def get_equity_df(members_df, contributions_df):
    """Gets a dataframe displaying each member's different types of equity, based on the sum
    of the amounts in the given list of contributions.
    """
    if len(contributions_df) == 0:
        equity_df = members_df.assign(membership=0,preferred=0,other=0,equity=0)
    else:
        equity_df = contributions_df.pivot_table(
            index=['name'], columns=['type'], values='amount', aggfunc=np.sum, fill_value=0)
        equity_df = members_df.merge(equity_df, left_on='name', right_index=True, how='outer')
        equity_df.fillna(0, inplace=True)

        equity_types = ['membership', 'preferred', 'other']
        for equity_type in equity_types:
            if equity_type not in equity_df:
                equity_df[equity_type] = 0

        equity_df['equity'] = equity_df[equity_types].sum(axis=1)

    return equity_df

def fraction_year_remaining(date, use_year_end=True):
    """Computes the fraction of the year remaining from a given date.
    """
    date = pd.to_datetime(date)
    if use_year_end:
        offset = pd.tseries.offsets.YearEnd()
    else:
        offset = pd.tseries.offsets.YearBegin()
    year = date - offset
    next_year = date + offset
    #print(offset, year, next_year)
    return ((next_year - date).days) / (next_year - year).days

def new_patronage_by_transaction(new_contributions_df):
    """Compute patronage for each transaction based on amount and fraction of year remaining.
    """
    patronage = new_contributions_df['amount']*new_contributions_df['date'].apply(fraction_year_remaining)
    return new_contributions_df.assign(patronage=patronage) #This makes a copy and adds a new column

def new_patronage_by_member(new_contributions_df):
    """Compute each member's patronage from new contributions for the current year.
    """
    new_contributions_df = new_patronage_by_transaction(new_contributions_df)
    return new_contributions_df[['member_id','patronage']].groupby(by='member_id').sum()

def compute_patronage(old_equity_df, new_contributions_df):
    """Compute total patronage for each member from new contributions for the current year
        and existing equity from previous years.
    """
    patronage_df = old_equity_df.set_index('member_id')[['name', 'equity']]
    patronage_df.rename(columns={'equity': 'old_patronage'}, inplace=True)

#     new_contributions_df = new_patronage_by_transaction(new_contributions_df)
#     new_contributions_df[['member_id','patronage']].groupby(by='member_id').sum()

    patronage_df['new_patronage'] = new_patronage_by_member(new_contributions_df)['patronage']
    # If there were members with no contributions this year, set their new patronage to 0 (would be NaN).
    patronage_df.fillna(0, inplace=True)

    patronage_df['patronage'] = patronage_df['old_patronage'] + patronage_df['new_patronage']
    patronage_df['proportionate_patronage'] = patronage_df['patronage'] / patronage_df['patronage'].sum()

    return patronage_df

def compute_patronage_for_year(members_df, contributions_df, year):
    """Computes each member's patronage for the specified year."""
    old_equity_df, new_contributions_df, _ = split_by_year(contributions_df, year)
    old_equity_df = get_equity_df(members_df, old_equity_df)
    return compute_patronage(old_equity_df, new_contributions_df)

def compute_dividends(patronage_df, profit, proportion_individual=0.5, rounded=True):
    """Compute each member's dividend based on patronage for the year.
    """
    dividend_df = patronage_df[['name', 'proportionate_patronage']].copy()

    #Compute individual patronage allocations
    dividend_df['dividend'] = dividend_df['proportionate_patronage'] * profit * proportion_individual
    if rounded:
        dividend_df['dividend'] = np.round(dividend_df['dividend'], 2)

    # To account for rounding amounts to the nearest cent, we add up the individual dividends
    # to get the actual amount allocated to individual net income. Then we subtract this amount
    # from the total profit to get the collective net income.
    indiv_profit = dividend_df['dividend'].sum()
    collective_profit = profit - indiv_profit

    # We reserve member_id=0 for the collective account (or we could simply use names as keys)
    dividend_df.loc[0] = pd.Series({
        'name': 'CollectiveAcct',
        'proportionate_patronage': collective_profit / indiv_profit,
        'dividend': collective_profit
    })
    return dividend_df

def compute_allocations(dividend_df,
                        year,
                        first_year_proportion=0.5,
                        n_years=3,
                        distribution = None,
                        irregular_payment='last'):
    """Computes allocations over next n_years years after the dividend year,
    or according to the given payment distribution.
    """

    #Currently we assume below that either all dividends will be positive or all will
    #be negative, depending on the overall profit.
    #Conceivably there could be a situation where some dividends could be positive
    #and some could be negative, in which case we'd need two distribution arrays,
    #but I can't currently think of why we would want to allow that.

    #If no explicit payment distribution was passed, we need to create one.
    if distribution is None:
        #Check whether the total profit is positive or negative, and create
        #the appropriate default distribution.
        if dividend_df['dividend'].sum() >= 0:
            #profit >= 0
            #Default for a positive dividend is to evenly evenly divide what's left over after
            #the first year over the remaining n-1 years.
            if n_years == 1:
                #Avoid divide by zero error.
                distribution = [1]
            else:
                #n_years > 1
                distribution = [(1-first_year_proportion) / (n_years-1) for _ in range(n_years)]
                distribution[0] = first_year_proportion
        else:
            #profit < 0
            #Default for a negative dividend is to evenly divide it over all n years.
            distribution = [1.0 / n_years for _ in range(n_years)]

    #Convert the distribution from a list to a numpy array to perform math with it.
    distribution = np.array(distribution)

    #In case the distribution was explicitly passed, make sure n_years matches the actual length.
    n_years = len(distribution)

    #Create a new DataFrame for the allocations by dropping the collective account and
    #the proportionate patronage column from dividend_df.
    allocation_df = dividend_df.drop(index=0, columns='proportionate_patronage')

    #Use broadcasting to compute all dividends with one multiplication,
    #and round them to nearest cent.
    allocations = (distribution.reshape(1,-1) * allocation_df['dividend'].values.reshape(-1,1)).round(2)

    #Adjust the first or last payment to account for rounding, by replacing the
    #first or last column, respectively, with the result of subtracting
    #the sum of the remaining payments from the member's actual dividend.
    #Or if they passed a specific year, adjust the payment for that year.
    if irregular_payment == 'first': irregular_payment = 1
    elif irregular_payment == 'last': irregular_payment = n_years
    elif 1 <= irregular_payment <= n_years: pass
    else: raise ValueError(
        "irregular_payment must be 'first', 'last', or an integer between 1 and n_years (inclusive).")

    sum_of_remaining = (allocations[:,:irregular_payment-1].sum(axis=1)
                            + allocations[:,irregular_payment:].sum(axis=1))
    allocations[:,irregular_payment-1] = np.round(allocation_df['dividend'] - sum_of_remaining, 2)

    #Create new column labels for the years year+1, year+2,...,year+n
    new_columns = [str(y) for y in range(year+1, year+n_years+1)]

    #Concatenate the existing allocation dataframes horizontally with the computed allocations.
    allocation_df = pd.concat([allocation_df,
                              pd.DataFrame(allocations, index=allocation_df.index, columns=new_columns)],
                             axis=1)

    #Rename the 'dividend' column.
    allocation_df.rename(columns={'dividend': str(year)+'_dividend'}, inplace=True)

    return allocation_df
