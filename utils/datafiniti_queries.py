
SOLD_HOMES_QUERY = '''\
    statuses.type:\'For Sale\'\
    AND statuses.type:Sold\
    AND prices.amountMin:*\
    AND prices.amountMax:*\
    AND features:*\
    AND descriptions.value:*\
    AND features.key:'\Property History\'
    AND propertyType:(Home OR \'Multi-Family Dwelling\' OR \'Single Family Dwelling\' OR Townhouse)\
    AND sourceURLs:https\://redfin.com\
    AND dateAdded:[2017-01-01 TO *]\
'''