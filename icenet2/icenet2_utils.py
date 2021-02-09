import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

###############################################################################
############### LOSS FUNCTIONS
###############################################################################

# TODO

###############################################################################
############### METRICS
###############################################################################

# TODO

###############################################################################
############### ARCHITECTURES
###############################################################################

# TODO

###############################################################################
############### MISC
###############################################################################

def filled_daily_dates(start_date, end_date):
    """
    Return a numpy array of datetimes, incrementing daily, starting at start_date and
    going up to (but not including) end_date.
    """

    monthly_list = []
    date = start_date

    while date < end_date:
        monthly_list.append(date)
        date += relativedelta(days=1)

    return np.array(monthly_list)
