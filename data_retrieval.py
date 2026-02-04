"""
Data retrieval functions for metrics and product data from ClickHouse with validation and error handling.
"""

import logging
from .utils import validate_dataframe


def validate_input_params(wid: str, start: str, end: str):
    """
    Validate input parameters for data retrieval functions.
    
    Args:
        wid: Website ID
        start: Start date in format 'YYYY-MM-DD'
        end: End date in format 'YYYY-MM-DD'
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not wid or not isinstance(wid, str):
        raise ValueError(f"Invalid website ID: {wid}")
    
    # Basic date format validation (more thorough validation could be added)
    if not start or not end:
        raise ValueError(f"Start and end dates are required: start={start}, end={end}")
    
    # Validate date format (basic check)
    if not (isinstance(start, str) and len(start) == 10 and start.count('-') == 2):
        raise ValueError(f"Invalid start date format: {start}. Expected YYYY-MM-DD")
    
    if not (isinstance(end, str) and len(end) == 10 and end.count('-') == 2):
        raise ValueError(f"Invalid end date format: {end}. Expected YYYY-MM-DD")


def get_metrics_by_country(client, wid, start, end): 
    """
    Retrieve metrics by country from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID
        start: Start date in format 'YYYY-MM-DD'
        end: End date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with metrics by country
    """
    validate_input_params(wid, start, end)
    
    query = f"""
    select * from raw_metrics_by_country FINAL where websiteId = '{wid}' and toDate(timestamp) between '{start}' and '{end}'
    """
    
    try:
        logging.info(f"Retrieving metrics by country for website {wid} from {start} to {end}")
        df_metrics_by_country = client.query_df(query)
        
        logging.info(f"Retrieved {len(df_metrics_by_country)} rows for metrics by country")
        return df_metrics_by_country
    except Exception as e:
        logging.error(f"Error retrieving metrics by country: {e}")
        raise


def get_metrics_by_product(client, wid, start, end):
    """
    Retrieve metrics by product from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID
        start: Start date in format 'YYYY-MM-DD'
        end: End date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with metrics by product
    """
    validate_input_params(wid, start, end)
    
    query = f"""
    select * from raw_metrics_by_product FINAL where websiteId = '{wid}' and toDate(timestamp) between '{start}' and '{end}'
    """
    
    try:
        logging.info(f"Retrieving metrics by product for website {wid} from {start} to {end}")
        df_metrics_by_product = client.query_df(query)
        
        logging.info(f"Retrieved {len(df_metrics_by_product)} rows for metrics by product")
        return df_metrics_by_product
    except Exception as e:
        logging.error(f"Error retrieving metrics by product: {e}")
        raise


def get_raw_metrics(client, wid, start, end):
    """
    Retrieve raw metrics from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID
        start: Start date in format 'YYYY-MM-DD'
        end: End date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with raw metrics
    """
    validate_input_params(wid, start, end)
    
    query = f"""
    select * from raw_metrics FINAL where websiteId = '{wid}' and toDate(timestamp) between '{start}' and '{end}'
    """
    
    try:
        logging.info(f"Retrieving raw metrics for website {wid} from {start} to {end}")
        df_metrics = client.query_df(query)
        
        logging.info(f"Retrieved {len(df_metrics)} rows for raw metrics")
        return df_metrics
    except Exception as e:
        logging.error(f"Error retrieving raw metrics: {e}")
        raise


def get_latest_ads(client, wid):
    """
    Retrieve latest ads data from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID

    Returns:
        DataFrame with latest ads data
    """
    if not wid or not isinstance(wid, str):
        raise ValueError(f"Invalid website ID: {wid}")
    
    # ads 
    query = f"""
    WITH latest_name_ad as (
        SELECT adId, 
            argMax(adName, insertedAt) as name
        from raw_metrics FINAL 
        where websiteId = '{wid}' and adName != ''
        group by 1
    ), 
    latest_name_adSet as (
        SELECT adSetId, 
            argMax(adSetName, insertedAt) as name
        from raw_metrics FINAL 
        where websiteId = '{wid}' and adName != ''
        group by 1
    ), 
    latest_name_campaign as (
        SELECT campaignId, 
            argMax(campaignName, insertedAt) as name
        from raw_metrics FINAL 
        where websiteId = '{wid}' and adName != ''
        group by 1
    )
    SELECT a.*, 
        latest_name_ad.name as adName, 
        latest_name_adSet.name as adSetName, 
        latest_name_campaign.name as campaignName
    from raw_ads as a 
        left join latest_name_ad on a.adId = latest_name_ad.adId
        left join latest_name_adSet on a.adSetId = latest_name_adSet.adSetId
        left join latest_name_campaign on a.campaignId = latest_name_campaign.campaignId
    where websiteId = '{wid}' settings final = 1
    """

    try:
        logging.info(f"Retrieving latest ads for website {wid}")
        df_ads = client.query_df(query)
        df_ads.columns = [c.split('.')[-1] for c in df_ads.columns] ## remove the prefixes
        
        logging.info(f"Retrieved {len(df_ads)} rows for latest ads")
        return df_ads
    except Exception as e:
        logging.error(f"Error retrieving latest ads: {e}")
        raise


def get_latest_products_and_groups(client, wid):
    """
    Retrieve latest products and groups data from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID

    Returns:
        DataFrame with latest products and groups data
    """
    if not wid or not isinstance(wid, str):
        raise ValueError(f"Invalid website ID: {wid}")
    
    # raw_products_new
    query = f"""
    SELECT productId, 
        argMax(productGroupId, updatedAt) as productGroupId, 
        argMax(productBundleId, updatedAt) as productBundleId, 
        argMax(status, updatedAt) as status, 
        argMax(name, updatedAt) as name, 
        argMax(description, updatedAt) as description, 
        argMax(sku, updatedAt) as sku, 
        argMax(stock, updatedAt) as stock, 
        argMax(onlineStock, updatedAt) as onlineStock, 
        argMax(inStoreStock, updatedAt) as inStoreStock, 
        argMax(stockStatus, updatedAt) as stockStatus, 
        argMax(price, updatedAt) as price, 
        argMax(compareAtPrice, updatedAt) as compareAtPrice, 
        argMax(cost, updatedAt) as cost, 
        argMax(imageUrl, updatedAt) as imageUrl, 
        argMax(url, updatedAt) as url, 
        argMax(brand, updatedAt) as brand, 
        argMax(barcode, updatedAt) as barcode, 
        argMax(metafieldsJSON, updatedAt) as metafieldsJSON, 
        argMax(weight, updatedAt) as weight, 
        argMax(isGiftCard, updatedAt) as isGiftCard,
        argMax(hasOnlyDefaultVariant, updatedAt) as hasOnlyDefaultVariant,
        argMax(productBundleQuantity, updatedAt) as productBundleQuantity,
        argMax(timezone, updatedAt) as timezone,
        argMax(publishedAt, updatedAt) as publishedAt        
        from raw_products_new FINAL 
    where websiteId = '{wid}'
    group by 1
    """
    
    try:
        logging.info(f"Retrieving latest products and groups for website {wid}")
        df_prod = client.query_df(query)
        
        logging.info(f"Retrieved {len(df_prod)} rows for latest products and groups")
        return df_prod
    except Exception as e:
        logging.error(f"Error retrieving latest products and groups: {e}")
        raise


def get_latest_product_groups(client, wid):
    """
    Retrieve latest product groups data from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID

    Returns:
        DataFrame with latest product groups data
    """
    if not wid or not isinstance(wid, str):
        raise ValueError(f"Invalid website ID: {wid}")
    
    # raw_product_groups_new
    query = f"""
    SELECT * from raw_product_groups_new FINAL
    where websiteId = '{wid}'
    """  
    
    try:
        logging.info(f"Retrieving latest product groups for website {wid}")
        df_prod_group = client.query_df(query)
        
        logging.info(f"Retrieved {len(df_prod_group)} rows for latest product groups")
        return df_prod_group
    except Exception as e:
        logging.error(f"Error retrieving latest product groups: {e}")
        raise


def get_websites(client): 
    """
    Retrieve websites data from ClickHouse.
    
    Args:
        client: ClickHouse client connection
        
    Returns:
        DataFrame with websites data
    """
    query = """select * from websites FINAL where status = True"""
    
    try:
        logging.info("Retrieving websites data")
        res = client.query_df(query)
        
        logging.info(f"Retrieved {len(res)} rows for websites")
        return res
    except Exception as e:
        logging.error(f"Error retrieving websites: {e}")
        raise