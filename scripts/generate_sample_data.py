#!/usr/bin/env python3
"""
Sample Data Generator for Ad-Product Attribution Pipeline

This script generates sample parquet files to test the pipeline end-to-end.

Usage:
    python scripts/generate_sample_data.py --output-dir ./data

Output:
    - data/ads.parquet
    - data/products.parquet
    - data/metrics.parquet
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def generate_sample_data(
    output_dir: str = "./data",
    num_products: int = 50,
    num_ads: int = 30,
    num_days: int = 30,
    seed: int = 42,
):
    """Generate sample ads, products, and metrics data."""
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    website_id = "sample_website_001"

    # ============================================================
    # Generate Products
    # ============================================================
    product_categories = ["Shoes", "Apparel", "Accessories", "Electronics", "Home"]
    product_types = {
        "Shoes": ["Running Shoe", "Casual Sneaker", "Hiking Boot", "Sandal", "Dress Shoe"],
        "Apparel": ["T-Shirt", "Hoodie", "Jacket", "Pants", "Shorts"],
        "Accessories": ["Watch", "Sunglasses", "Belt", "Wallet", "Backpack"],
        "Electronics": ["Headphones", "Speaker", "Charger", "Cable", "Power Bank"],
        "Home": ["Lamp", "Pillow", "Blanket", "Mug", "Frame"],
    }
    colors = ["Black", "White", "Blue", "Red", "Green", "Gray", "Navy"]

    products = []
    for i in range(num_products):
        category = random.choice(product_categories)
        product_type = random.choice(product_types[category])
        color = random.choice(colors)

        product_name = f"{color} {product_type}"
        product_slug = product_name.lower().replace(" ", "-")

        products.append({
            "websiteId": website_id,
            "productGroupId": f"pg_{i:04d}",
            "productId": f"prod_{i:04d}",
            "name": product_name,
            "productGroupName": category,
            "url": f"https://example.com/products/{product_slug}",
            "collections": category,
        })

    products_df = pd.DataFrame(products)
    products_path = os.path.join(output_dir, "products.parquet")
    products_df.to_parquet(products_path, index=False)
    print(f"Generated {len(products_df)} products -> {products_path}")

    # ============================================================
    # Generate Ads
    # ============================================================
    platforms = ["google", "meta", "tiktok"]
    campaign_types = ["Brand", "Performance", "Retargeting", "Awareness"]

    ads = []
    for i in range(num_ads):
        platform = random.choice(platforms)
        campaign_type = random.choice(campaign_types)

        # Some ads target specific products (for matching)
        if random.random() < 0.6:  # 60% of ads target specific products
            target_product = random.choice(products)
            destination_url = target_product["url"]
            ad_name = f"{campaign_type} - {target_product['name']}"
        else:  # Generic ads
            destination_url = f"https://example.com/collections/{random.choice(product_categories).lower()}"
            ad_name = f"{campaign_type} - {random.choice(product_categories)} Collection"

        campaign_id = f"camp_{i // 3:03d}"
        adset_id = f"adset_{i // 2:03d}"

        ads.append({
            "websiteId": website_id,
            "platform": platform,
            "campaignId": campaign_id,
            "campaignName": f"{campaign_type} Campaign {i // 3 + 1}",
            "adSetId": adset_id,
            "adSetName": f"{platform.title()} Audience {i // 2 + 1}",
            "adId": f"ad_{i:04d}",
            "name": ad_name,
            "destinationUrl": destination_url,
        })

    ads_df = pd.DataFrame(ads)
    ads_path = os.path.join(output_dir, "ads.parquet")
    ads_df.to_parquet(ads_path, index=False)
    print(f"Generated {len(ads_df)} ads -> {ads_path}")

    # ============================================================
    # Generate Metrics
    # ============================================================
    start_date = datetime.now() - timedelta(days=num_days)

    metrics = []
    for day in range(num_days):
        current_date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")

        for ad in ads:
            # Each ad has metrics for 1-5 products per day
            num_products_for_ad = random.randint(1, 5)
            selected_products = random.sample(products, min(num_products_for_ad, len(products)))

            # Generate base spend/impressions for this ad
            base_spend = random.uniform(10, 500)
            base_impressions = int(base_spend * random.uniform(50, 200))

            for j, product in enumerate(selected_products):
                # First product gets more weight (simulates lead product)
                weight = 0.5 if j == 0 else random.uniform(0.1, 0.3)

                spend = base_spend * weight
                impressions = int(base_impressions * weight)
                clicks = int(impressions * random.uniform(0.01, 0.05))
                conversions = int(clicks * random.uniform(0.02, 0.15))
                gross_profit = conversions * random.uniform(20, 100)

                metrics.append({
                    "date": current_date,
                    "websiteId": website_id,
                    "platform": ad["platform"],
                    "campaignId": ad["campaignId"],
                    "campaignName": ad["campaignName"],
                    "adSetId": ad["adSetId"],
                    "adSetName": ad["adSetName"],
                    "adId": ad["adId"],
                    "adName": ad["name"],
                    "productGroupId": product["productGroupId"],
                    "productId": product["productId"],
                    "productGroupName": product["productGroupName"],
                    "productName": product["name"],
                    "spend": round(spend, 2),
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "grossProfit": round(gross_profit, 2),
                })

    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "metrics.parquet")
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"Generated {len(metrics_df)} metric rows -> {metrics_path}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SAMPLE DATA GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - ads.parquet      ({len(ads_df)} ads)")
    print(f"  - products.parquet ({len(products_df)} products)")
    print(f"  - metrics.parquet  ({len(metrics_df)} rows, {num_days} days)")
    print(f"\nNext steps:")
    print(f"  1. Run the pipeline:")
    print(f"     python -m simple.run_pipeline --from-parquet {output_dir} --output-dir ./results --format both")
    print(f"")
    print(f"  2. Start the dashboard:")
    print(f"     streamlit run app.py")
    print(f"")
    print(f"  3. Upload results/sku_allocation.csv or .parquet in the dashboard")

    return ads_df, products_df, metrics_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample data for the Ad-Product Attribution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default sample data:
  python scripts/generate_sample_data.py --output-dir ./data

  # Generate larger dataset:
  python scripts/generate_sample_data.py --output-dir ./data --num-products 100 --num-ads 50 --num-days 60
        """
    )

    parser.add_argument("--output-dir", "-o", default="./data",
                        help="Output directory for generated files")
    parser.add_argument("--num-products", type=int, default=50,
                        help="Number of products to generate")
    parser.add_argument("--num-ads", type=int, default=30,
                        help="Number of ads to generate")
    parser.add_argument("--num-days", type=int, default=30,
                        help="Number of days of metrics data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    generate_sample_data(
        output_dir=args.output_dir,
        num_products=args.num_products,
        num_ads=args.num_ads,
        num_days=args.num_days,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
