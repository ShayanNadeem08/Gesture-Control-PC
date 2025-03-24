from time import gmtime, strftime
def get():
    timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    return timestamp
