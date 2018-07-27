from datetime import datetime


def get_today_date(format_="%Y%m%d"):
    today = datetime.today()
    return datetime.strftime(today, format_)
