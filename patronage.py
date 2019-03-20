"""
Module for calculating patronage, dividends, and written notices of allocation.

The module is designed in a functional style.
You start with 4 dataframes, and peform successive transformations
on them by passing the output dataframes into another function.

contributions_df = get_contributions_df(members_df, membership_df, preferred_df, other_equity_df)
old_patronage_df, new_patronage_df = old_new_patronage_for_year(contributions_df, year)
patronage_df = total_patronage_for_members(members_df, old_patronage_df, new_patronage_df)
dividend_df = compute_dividends(patronage_df, profit)
allocation_df = compute_allocations(dividend_df, year)
"""


import numpy as np
import pandas as pd
import date_utils

COLLECTIVE_ACCT_ID = 0
INDIV_ACCTS_ID = 9999
TOTAL_PROFIT_ID = 10000

def load_members_df(filepath='input/members.csv'):
    """Load and return the members DataFrame from the specified .csv file"""
    return pd.read_csv(filepath)

def load_equity_dfs(membership_equity_path='input/membership-equity.csv',
                    preferred_equity_path='input/preferred-equity.csv',
                    other_equity_path='input/other-equity.csv'):
    """Load and return the equity DataFrames from the specified .csv files"""
    membership_df = pd.read_csv(membership_equity_path)
    preferred_df = pd.read_csv(preferred_equity_path)
    other_equity_df = pd.read_csv(other_equity_path)
    return membership_df, preferred_df, other_equity_df

def get_contributions_df(members_df, membership_df, preferred_df, other_equity_df):
    """Combines data from all four data tables into one list of transactions.
    """
    #This code would modify the input dataframes:
    #membership_df['type'] = 'membership'
    #preferred_df['type'] = 'preferred'
    #other_equity_df['type'] = 'other'

    #Instead, use assign to make a copy of the dataframes rather than modifying them in place:
    membership_df = membership_df.assign(type='membership')
    preferred_df = preferred_df.assign(type='preferred')
    other_equity_df = other_equity_df.assign(type='other')

    equity_dfs = [membership_df, preferred_df, other_equity_df]
    #Doing an inner join ignores superfluous columns in the input dataframes
    contributions_df = pd.concat(equity_dfs, join='inner', ignore_index=True)

    contributions_df = members_df.merge(contributions_df, on='name')
    #contributions_df.rename(columns={'id': 'member_#'}, inplace=True)

    return contributions_df

def split_by_year(contributions_df, year_to_split_on, return_future=False):
    """Splits a list of transactions those occurring before, during,
    and (optionally) after the given year.
    """
    year = pd.to_datetime(contributions_df['date']).apply(lambda date: date.year)
    before = contributions_df[year < year_to_split_on]
    during = contributions_df[year == year_to_split_on]
    if return_future:
        return before, during, contributions_df[year > year_to_split_on]
    else:
        return before, during

def prepend_year(year, *col_names):
    """Prepend a year to a string or sequence of strings."""
    new_names = [str(year) + "_" + col_name for col_name in col_names]
    return new_names[0] if len(new_names) == 1 else new_names

def get_year_from_names(*names):
    """Tests whether a name or list of names share a common 4-digit number (e.g. year) prepended to them.
    If so, returns the year as an int, otherwise (no year or years don't match) returns None.
    """
    # #If a single name was passed, put it in a list.
    # if type(names) is str: names = [names]

    #Get pairs of consecutive elements of the list.
    consecutive_pairs = zip(names[:len(names)-1], names[1:])

    #Check whether the first 4 digits of the names in each pair match.
    #If so, all names share a common prefix.
    #Note: all([]) evaluates to True, so this still works if len(names) == 1.
    if all(name1[:4] == name2[:4] for name1, name2 in consecutive_pairs):
        #Check if the matching prefix is a digit, and if so return it.
        if names[0][:4].isdigit():
            return int(names[0][:4])

    #If there was a mismatch or the prefix wasn't an integer, return None.
    return None

def is_match(name, col_name):
    """Checks whether col_name either equals name or equals name prepended by a 4-digit number
    plus one character (e.g. year plus a space or underscore).
    """
    return (col_name[-len(name):] == name) and (
        len(col_name) == len(name) or
        (len(col_name) == len(name)+5 and col_name[:4].isdigit())
    )

def find_matching_names(columns, *names):
    """Finds the actual column names in columns that either match the names
     exactly or have the year prepended.

    Examples (old version, passing two lists):
    In: find_matching_names(names=['yak', 'snack'],
            columns=['grundle', 'yak', 'smurf', 'glork', '3401 snack', 'glurg'])
    Out: ['yak', '3401 snack']

    In: find_matching_names(names=['yak', 'snack'],
            columns=['grundle', 'snack', 'yak', 'smurf', 'glork', '3401 snack', 'glurg'])
    Out: ['yak', 'snack', '3401 snack']

    In: find_matching_names(names='snack',
        columns=['grundle', 'snack', 'yak', 'smurf', 'glork', '3401 snack', 'glurg'])
    Out: ['snack', '3401 snack']

    New version (with argument packing for names):
    In: find_matching_names(['2222_yak', 'snack', 'grundle', 'yak', 'smurf', 'glork', '3401 snack', 'glurg'],
                      'yak', 'snack', 'goat')
    Out: ['2222_yak', 'yak', 'snack', '3401 snack']
    """

    # #If a single string was passed, replace it with a list
    # if type(names) == str: names = [names]
    # if type(columns) == str: columns = [columns]

    return [col for name in names for col in columns if is_match(name, col)]



def old_patronage(old_contributions_df):
    """Compute the "old patronage" given all the transactions before a given year.
    """
    id_columns = [c for c in old_contributions_df.columns if c in ['member_#', 'name']]
    patronage_types = ['membership', 'preferred', 'other']
    old_patronage_column = 'old_patronage'

    #indiv_accts_row = {'member_#': INDIV_ACCTS_ID, 'name': 'All Individual Accts.'}
    #id_columns = ['member_#']

    if len(old_contributions_df) == 0:
        #An empty old_contributions DataFrame needs to be handled separately
        #to avoid an error when calling df.pivot_table.
        #Create an empty dataframe with the correct columns for old patronage.
        old_patronage_df = pd.DataFrame(columns=id_columns + patronage_types)
    else:
        #Pivot the old_contributions dataframe, using the different types of patronage as the new columns,
        #with 'member_#' and/or 'name' as the index.
        old_patronage_df = old_contributions_df.pivot_table(
            index=id_columns, columns=['type'], values='amount', aggfunc=np.sum, fill_value=0)

        #Move 'member_#' and/or 'name' into columns rather than keeping them as the index.
        old_patronage_df.reset_index(inplace=True)
        #Remove the name 'type' from the columns axis.
        old_patronage_df.rename_axis(None, axis='columns', inplace=True)

        #If some of the types of patronage didn't occur, explicitly add 0 patronage for that type.
        for patronage_type in patronage_types:
            if patronage_type not in old_patronage_df:
                old_patronage_df[patronage_type] = 0

    # #Original version (2018-10-27)
    # #Add a row for the total of all individual accounts
    # indiv_accts_row = old_patronage_df[patronage_types].sum(axis=0).fillna(0)
    # #If we started with an empty DataFrame, things will be floats by default.
    # #Change the type to int so that the 'member_#' column remains integer-typed.
    # if all(indiv_accts_row == 0):
    #     indiv_accts_row = indiv_accts_row.astype('int')
    # if 'member_#' in id_columns:
    #     indiv_accts_row['member_#'] = INDIV_ACCTS_ID
    # if 'name' in id_columns:
    #     indiv_accts_row['name'] = 'All Individual Accts.'
    #
    # old_patronage_df = old_patronage_df.append(indiv_accts_row, ignore_index=True)

    #Second attempt (2018-01-28)
    #Add a row for the total of all individual accounts
    #If the list of columns is missing one of 'name' or 'member_#',
    #this field in the dictionary will be ignored when the DataFrame
    #is created. The remaining columns will be filled with NaN.
    indiv_accts_row = pd.DataFrame({'member_#': INDIV_ACCTS_ID,
                                    'name': 'All Individual Accts.'
                                    }, index=[0], columns=id_columns + patronage_types)
    #Calling .sum(axis=0) on the sub-DataFrame returns a Series whose
    #index matches the columns we want to fill. We need to transpose this,
    #so we convert it to a DataFrame and call .T for transpose.
    #Calling .fillna(0) is necessary in case the passed old_contributions_df
    #DataFrame is empty, in which case .sum() results in NaNs. Perhaps
    #setting initial=0 would work instead, but I need to update numpy to try it.
    indiv_accts_row[patronage_types] = (
            old_patronage_df[patronage_types]
            .sum(axis=0).fillna(0).values
            )
    #Append the new row. When I upgrade pandas, set sort=False.
    old_patronage_df = old_patronage_df.append(indiv_accts_row, ignore_index=True)
    #old_patronage_df = old_patronage_df.merge(indiv_accts_row, how='outer')

    #reorder the columns in the desired order(??), and compute the total old patronage
    #by adding up the different types and storing the value in a new column.
    old_patronage_df[old_patronage_column] = old_patronage_df[patronage_types].sum(axis=1)

    return old_patronage_df

def new_patronage(new_contributions_df, year_fraction_options=None):
    """Compute the "new patronage" given all the transactions that occurred during a given year."""
    if year_fraction_options is None:
        year_fraction_options = dict()
    #Passing year_fraction_options={} uses the default options for fraction_year_remaining.
    #Equivalently, year_fraction_options=dict(use_year_end=True, method='actual/actual')
    year_fraction_function = lambda date: date_utils.fraction_year_remaining(date, **year_fraction_options)
    year_fraction = new_contributions_df['date'].apply(year_fraction_function)
    new_patronage = new_contributions_df['amount']*year_fraction
    #This makes a copy and adds new columns
    new_patronage_df = new_contributions_df.assign(
        year_fraction=year_fraction, new_patronage=new_patronage)

    #Add a row for the total of all individual accounts.
    #Use same procedure as in old_patronage function (see comments there).
    indiv_accts_row = pd.DataFrame({'member_#': INDIV_ACCTS_ID,
                                   'name': 'All Individual Accts.'
                                  }, index=[0], columns=new_patronage_df.columns)
    indiv_accts_row[['amount','new_patronage']] = (
            new_patronage_df[['amount','new_patronage']]
            .sum(axis=0).fillna(0).values
            )
    #Assign the weighted average of all the year fraction to the summary account
    indiv_accts_row['year_fraction'] = indiv_accts_row['new_patronage'] / indiv_accts_row['amount']

    #Append the individual accounts row to the dataframe
    new_patronage_df = new_patronage_df.append(indiv_accts_row, ignore_index=True)
    #print(new_patronage_df)

    totals = new_patronage_df[['member_#','new_patronage']].groupby(by='member_#', as_index=False).sum()
    #print(totals)
    #totals.reset_index(inplace=True)
    totals.rename(columns={'new_patronage': 'member_total'}, inplace=True)
    return new_patronage_df.merge(totals, on='member_#')

def old_new_patronage_for_year(contributions_df, year, year_fraction_options=None):
    """Return both the old and new patronage for a given year.
    The 'old_patronage' and 'new_patronage' columns will be prepended by the year.
    """
    old_contributions_df, new_contributions_df = split_by_year(contributions_df, year)[:2]
    old_patronage_df = old_patronage(old_contributions_df)
    new_patronage_df = new_patronage(new_contributions_df, year_fraction_options)

    #Now rename columns with year
    old_patronage_col, new_patronage_col = prepend_year(year, 'old_patronage', 'new_patronage')
    old_patronage_df.rename(columns={'old_patronage': old_patronage_col}, inplace=True)
    new_patronage_df.rename(columns={'new_patronage': new_patronage_col}, inplace=True)

    return old_patronage_df, new_patronage_df

def total_patronage_for_members(members_df, old_patronage_df, new_patronage_df, join_columns=['member_#','name']):
    """Computes the total patronage for the specified members given lists of old patronage and new patronage.
    """
    #Since find_matching_names returns a list, we're assigning to a tuple, so we need a comma
    #after the variable name to denote a tuple of length 1.
    old_patronage_col, = find_matching_names(old_patronage_df.columns, 'old_patronage')
    new_patronage_col, = find_matching_names(new_patronage_df.columns, 'new_patronage')
    patronage_col = 'patronage'

    #Add "All Individual Accts." as a member, to compute totals for all accounts.
    members_df = members_df.append({'member_#': INDIV_ACCTS_ID, 'name': 'All Individual Accts.'}, ignore_index=True)

    #Keep member_# and/or name only if the column is in both patronage dataframes.
    #This will cause an error if the two patronage dataframes don't share a common column.
    # join_columns = [c for c in old_patronage_df.columns if c in ['member_#','name']]
    # join_columns = [c for c in new_patronage_df.columns if c in join_columns]
    join_columns = list(set(join_columns).intersection(old_patronage_df.columns, new_patronage_df.columns))

    #Join the members dataframe with the old and new patronage using the join columns.
    #Use left joins to compute patronage precisely for the members in members_df.
    #We drop duplicates in new_patronage_df because members with more than one
    #contribution during the year will show up more than once.
    patronage_df = (members_df[join_columns]
                    .merge(old_patronage_df[join_columns + [old_patronage_col]],
                        on=join_columns, how='left')
                    .merge(new_patronage_df[join_columns + ['member_total']].drop_duplicates(),
                        on=join_columns, how='left')
                   )

    #Rename the 'member_total' column as new_patronage_col.
    patronage_df.rename(columns={'member_total': new_patronage_col}, inplace=True)

    #If there were members with no patronage, set their patronage to 0.
    #(otherwise we'd get NaN's)
    patronage_df.fillna(0, inplace=True)

    #Check whether the old and new patronage columns share a comon prepended year.
    #If so, prepend the year to the patronage column.
    year = get_year_from_names(old_patronage_col, new_patronage_col)
    if year is not None:
        patronage_col = prepend_year(year, patronage_col)

    #Compute the total patronage and proportionate patronage and store them to new columns.
    patronage_df[patronage_col] = patronage_df[old_patronage_col] + patronage_df[new_patronage_col]
    patronage_df['proportionate_patronage'] = (
        patronage_df[patronage_col] / patronage_df.loc
            [patronage_df['member_#'] != INDIV_ACCTS_ID, patronage_col].sum()
            )

    return patronage_df

# def compute_patronage_for_year(members_df, contributions_df, year):
#     """Computes each member's patronage for the specified year."""
#     old_equity_df, new_contributions_df = split_by_year(contributions_df, year)[:2]
#     old_equity_df = get_equity_df(members_df, old_equity_df)
#     return compute_patronage(old_equity_df, new_contributions_df)



def compute_dividends(patronage_df, profit, proportion_individual=0.5, rounded=True):
    """Compute each member's dividend based on patronage for the year.
    """
    # #Create a new dataframe by adding 3 rows to the end of patronage_df
    # dividend_df = patronage_df.append(
    #     pd.DataFrame({'member_#': [COLLECTIVE_ACCT_ID, INDIV_ACCTS_ID, TOTAL_PROFIT_ID],
    #                   'name': ['Collective Acct.','Individual Accts.','Total Profit']}
    #                   ), ignore_index=True, sort=False) #Oops, need pandas 0.23.0 to use sort

    #Drop the 'All Individual Accts.' row, and add it back in later.
    #This just makes the code simpler than trying to keep it and do
    #something different for this vs. the other two rows to add.
    patronage_df = patronage_df.loc[patronage_df['member_#']!=INDIV_ACCTS_ID]

    #Create a new dataframe by adding 3 rows to the end of patronage_df.
    #Use outer join because we want to keep all data in both dataframes.
    #Leaving the join columns unspecified (on=None) defaults to joining on
    #the intersection of the columns in both dataframes (i.e. member_#, name,
    #or both). If either 'member_#' or 'name' is missing from patronage_df,
    #a new column will be created.
    dividend_df = patronage_df.merge(
        pd.DataFrame({'member_#': [INDIV_ACCTS_ID, COLLECTIVE_ACCT_ID, TOTAL_PROFIT_ID],
                      'name': ['All Individual Accts.', 'Collective Acct.','Total Profit']}
                      ), how='outer')

    #dividend_df.loc[dividend_df['name']=='Individual Accts.', 'proportionate_patronage'] = 1.0

    #Compute the proportion of profit for each member, and save to a new column
    dividend_df['proportion_of_profit'] = dividend_df['proportionate_patronage'] * proportion_individual

    #Create name for dividend column - prepend 'dividend' with year if one is found in patronage_df
    dividend_col = 'dividend'
    year = get_year_from_names(*find_matching_names(patronage_df.columns, 'patronage'))
    if year is not None:
        dividend_col = prepend_year(year, dividend_col)

    #Compute individual patronage allocations
    dividend_df[dividend_col] = dividend_df['proportion_of_profit'] * profit
    if rounded:
        dividend_df[dividend_col] = np.round(dividend_df[dividend_col], 2)

    #Find all 3 patronage columns and compute the sum of patronage for all individual accounts.
    patronage_columns = find_matching_names(patronage_df.columns, 'old_patronage', 'new_patronage', 'patronage')
    dividend_df.loc[dividend_df['member_#']==INDIV_ACCTS_ID, patronage_columns] = (
        patronage_df[patronage_columns].sum(axis=0).values
    )

    # To account for rounding amounts to the nearest cent, we add up the individual dividends
    # to get the actual amount allocated to individual net income. Then we subtract this amount
    # from the total profit to get the collective net income.
    # sum_dividends = dividend_df[dividend_col].sum().round(2) #This could be different
    indiv_profit = dividend_df[dividend_col].round(2).sum().round(2) #post-rounding should be unnecessary except for floating point differences
    collective_profit = np.round(profit - indiv_profit,2)

    # # We reserve member_#=0 for the collective account (or we could simply use names as keys)
    # dividend_df.loc[0] = pd.Series({
    #     'name': 'Collective Acct.',
    #     'proportionate_patronage': collective_profit / indiv_profit,
    #     'proportion_of_profit': collective_profit / profit,
    #     'dividend': collective_profit
    # })
    # dividend_df.loc[-1] = pd.Series({
    #     'name': 'Individual Accts.',
    #     'proportionate_patronage': 1.0,
    #     'proportion_of_profit': indiv_profit / profit,
    #     'dividend': indiv_profit
    # })
    # dividend_df.loc[-2] = pd.Series({
    #     'name': 'Total Profit',
    #     'proportionate_patronage': profit / indiv_profit,
    #     'proportion_of_profit': 1.0,
    #     'dividend': profit
    # })


    final_columns = ['proportionate_patronage', 'proportion_of_profit', dividend_col]
    collective_vals = [collective_profit / indiv_profit, collective_profit / profit, collective_profit]
    indiv_vals = [1.0, indiv_profit / profit, indiv_profit]
    total_vals = [profit / indiv_profit, 1.0, profit]

    dividend_df.loc[dividend_df['member_#']==COLLECTIVE_ACCT_ID, final_columns] = collective_vals
    dividend_df.loc[dividend_df['member_#']==INDIV_ACCTS_ID, final_columns] = indiv_vals
    dividend_df.loc[dividend_df['member_#']==TOTAL_PROFIT_ID, final_columns] = total_vals

    # print(f'sum_dividends: {sum_dividends}, indiv_profit: {indiv_profit}, collective_profit: {collective_profit}')

    return dividend_df

def dividend_calculations(patronage_df, dividend_df):
    """Merges the patronage and dividend dataframes to return a dataframe
    with all the patronage and dividend calculations for the year.
    """
    return patronage_df.reset_index().merge(
        dividend_df.reset_index(),on=None, how='outer'
        ).set_index('member_#')

def compute_allocations(dividend_df,
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

    columns_to_keep = find_matching_names(dividend_df.columns, 'member_#', 'name', 'dividend')
    #dividend_col, = find_matching_names(dividend_df.columns, 'dividend')
    *id_columns, dividend_col = columns_to_keep
    #print(dividend_col)
    year = get_year_from_names(dividend_col)

    #If no explicit payment distribution was passed, we need to create one.
    if distribution is None:
        #Check whether the total profit is positive or negative, and create
        #the appropriate default distribution.
        if dividend_df[dividend_col].sum() >= 0:
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
    #allocation_df = dividend_df.drop(columns='proportionate_patronage')
    allocation_df = dividend_df[columns_to_keep]

    #Use broadcasting to compute all dividends with one multiplication,
    #and round them to nearest cent.
    allocations = (distribution.reshape(1,-1) * allocation_df[dividend_col].values.reshape(-1,1)).round(2)

    #Adjust the first or last payment to account for rounding, by replacing the
    #first or last column, respectively, with the result of subtracting
    #the sum of the remaining payments from the member's actual dividend.
    #Or if they passed a specific year, adjust the payment for that year.
    if irregular_payment == 'first': irregular_payment = 1
    elif irregular_payment == 'last': irregular_payment = n_years
    elif not (1 <= irregular_payment <= n_years):
        raise ValueError(
        "irregular_payment must be 'first', 'last', or an integer between 1 and n_years (inclusive).")

    #This adjustment works for any irregular payment year in the range 1 to n_years.
    #Note that the index of the irregular payment is the one less
    #than the year number.
    sum_of_remaining = (allocations[:,:irregular_payment-1].sum(axis=1)
                            + allocations[:,irregular_payment:].sum(axis=1))
    allocations[:,irregular_payment-1] = np.round(allocation_df[dividend_col] - sum_of_remaining, 2)

    #Create new column labels for the years year+1, year+2,...,year+n
    if year is None:
        new_columns = [f'year_{y}' for y in range(1, n_years+1)]
    else:
        new_columns = [str(y) for y in range(year+1, year+n_years+1)]

    #Concatenate the existing allocation dataframes horizontally with the computed allocations.
    allocation_df = pd.concat([allocation_df,
                              pd.DataFrame(allocations, index=allocation_df.index, columns=new_columns)],
                             axis=1)

    # #Rename the 'dividend' column.
    # allocation_df.rename(columns={'dividend': str(year)+'_dividend'}, inplace=True)

    return allocation_df

def get_years_due(allocation_df):
    """Takes a yearly allocation dataframe and returns the years in which
    allocations issued that year are due.
    """
    return sorted(int(s) for s in allocation_df.columns if s.isdigit())

def unpivot_allocations(allocation_df):
    """Return a dataframe listing the allocations in allocation_df, in "unpivoted" form.
    In this version, years are stored as integers.
    """
    years_due = get_years_due(allocation_df)
    year_issued = years_due[0]-1

    dividend_col = str(year_issued) + '_dividend'

    allocations_by_year_due_dfs = (
            [(allocation_df.loc[allocation_df[dividend_col] > 0, ['name', str(y)]]
            .rename(columns={str(y): 'amount'})
            .assign(year_issued=year_issued, year_due=y, year_paid=np.NaN))
           for y in years_due]
           )

    #Vertically concatenate the dataframes for different years due
    return pd.concat(allocations_by_year_due_dfs).reset_index()

def unpivot_allocations_with_melt(allocation_df):
    """Return a dataframe listing the allocations in allocation_df, in "unpivoted" form,
    using the pandas melt function.
    In this version, years are stored as strings."""
    years_due = get_years_due(allocation_df)
    year_issued = years_due[0]-1

    years_due_cols = [str(y) for y in years_due]
    dividend_col = str(year_issued) + '_dividend'

    #Move member_# from index to a column
    allocation_df = allocation_df.reset_index()

    unpivoted_df = allocation_df.loc[allocation_df[dividend_col] > 0,:].melt(
    id_vars = ['member_#','name'], value_vars = years_due_cols, var_name = 'year_due', value_name='amount')

    unpivoted_df['year_issued'] = str(year_issued)
    unpivoted_df['year_paid'] = np.NaN

    #Reorder the columns on return
    return unpivoted_df[['member_#','name','amount','year_issued','year_due','year_paid']]

def list_all_allocations(allocation_dfs):
    """Takes a list of yearly allocation dataframes, unpivots them, and
    concatenates them into one long list of allocations.
    """
    return pd.concat([unpivot_allocations(df) for df in allocation_dfs],
        ignore_index=True).rename_axis('notice_#', axis='index', inplace=True)

def create_parameters_df():
    index = pd.Series([2016, 2017, 2018], name='years')

    tuples = [('patronage', 'year_fraction_options'),
              ('dividends', 'profit'),
              ('dividends', 'proportion_individual'),
              ('dividends', 'rounded'),
              ('allocations', 'first_year_proportion'),
              ('allocations', 'n_years'),
              ('allocations', 'distribution'),
              ('allocations', 'irregular_payment')
             ]
    columns = pd.MultiIndex.from_tuples(tuples, names=['function', 'parameters'])

    parameters = pd.DataFrame({('patronage', 'year_fraction_options'): [None, None, None],
                            ('dividends', 'profit'): [678.49, 1981.18, 851.73],
                               ('dividends', 'proportion_individual'): [0.5, 0.5, 0.5],
                            ('dividends', 'rounded'): [True, False, True],
                               ('allocations', 'first_year_proportion'): [0.5, 0.5, 0.5],
                            ('allocations', 'n_years'): [2,3,3],
                               ('allocations', 'distribution'): [None, None, None],
                            ('allocations', 'irregular_payment'): ['last', 'last', 'last']
                           }, index=index, columns=columns)
    return parameters
