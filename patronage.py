import numpy as np
import pandas as pd

def fraction_year_remaining(date):
    """Computes the fraction of the year remaining from a given date.
    """
    date = pd.to_datetime(date)
    offset = pd.tseries.offsets.YearBegin()
    year = date - offset
    next_year = date + offset
    #print(offset, year, next_year)
    return (next_year - date).days / (next_year - year).days

def compute_transaction_patronage(contributions_df):
    """Compute patronage for each transaction based on amount and fraction of year remaining.
    """
    contributions_df = contributions_df.copy()
    patronage = contributions_df['amount']*contributions_df['date'].apply(fraction_year_remaining)
    contributions_df['patronage'] = patronage
    return contributions_df

def compute_new_patronage(contributions_df):
    """Compute each member's patronage from new contributions for the current year.
    """
    contributions_df = compute_transaction_patronage(contributions_df)
    return contributions_df[['member_id','patronage']].groupby(by='member_id').sum()

def compute_patronage(prev_equity_df, contributions_df):
    """Compute total patronage for each member from new contributions for the current year
        and existing equity from previous years.
    """
    patronage_df = prev_equity_df.set_index('member_id')

    patronage_df['old_patronage'] = patronage_df['equity'] + patronage_df['preferred']
    patronage_df['new_patronage'] = compute_new_patronage(contributions_df)['patronage']
    patronage_df['patronage'] = patronage_df['old_patronage'] + patronage_df['new_patronage']

    patronage_df['proportionate_patronage'] = patronage_df['patronage'] / patronage_df['patronage'].sum()

    return patronage_df[['name', 'old_patronage', 'new_patronage', 'patronage', 'proportionate_patronage']]

def compute_dividends(patronage_df, profit, percent_individual=50):
    """Compute each member's dividend based on patronage for the year.
    """
#     indiv_profit, collective_profit = compute_indiv_collective_profit(profit, percent_individual)
    dividend_df = patronage_df[['name', 'proportionate_patronage']].copy()

    #Compute individual patronage allocations
    dividend_df['dividend'] = np.round(
        dividend_df['proportionate_patronage'] * profit * percent_individual/100, 2)

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

def compute_allocations(dividend_df, year, payout_percent=50, n_years=2):
    """Computes payout for current year and allocation over next n_years years.
    """
    allocation_df = dividend_df.drop(index=0, columns='proportionate_patronage')
    notice_amounts = dividend_df['dividend'].apply(
        lambda dividend: np.round(dividend*payout_percent*0.01/n_years,2))

    allocation_df[str(year+1)] = np.round(dividend_df['dividend'] - n_years*notice_amounts, 2)

    for y in range(year+2, year+2+n_years):
        allocation_df[str(y)] = notice_amounts

    allocation_df.rename(columns={'dividend': str(year)+'_dividend'}, inplace=True)

    return allocation_df
