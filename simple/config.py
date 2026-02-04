"""
Simplified Configuration for Ad-to-Product Attribution Pipeline.

Edit these values to customize the pipeline behavior.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date


@dataclass
class PipelineConfig:
    """
    Main configuration for the ad-product attribution pipeline.

    Key concepts:
    - Lead products: Products explicitly targeted by an ad (via URL, name match)
    - Halo products: Other products that sell as a spillover effect
    - Fair allocation: Distributes ad spend/impressions proportionally to all products
    - Lead-only allocation: Distributes only to lead products
    """

    # === Data Source ===
    website_id: str = "6839260124a2adf314674a5e"
    start_date: str = "2025-10-01"
    end_date: str = "2025-12-31"

    # === ClickHouse Connection ===
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "profitpeak"
    use_aws_secrets: bool = True  # If True, fetch creds from AWS Secrets Manager
    aws_profile: str = "live"
    aws_secret_name: str = "SHARED_LAMBDA_CREDENTIALS"
    aws_region: str = "ap-southeast-2"

    # === Matching Thresholds ===
    fuzzy_threshold: int = 85      # Min fuzzy match score (0-100). Higher = stricter.
    fuzzy_limit: int = 30          # Max fuzzy matches to consider per query
    min_token_length: int = 3      # Min chars for token-based matching
    max_products_per_ad: int = 50  # Cap on products matched per ad

    # === Output ===
    output_dir: str = "./results"
    output_formats: List[str] = field(default_factory=lambda: ["csv", "parquet"])

    # === Processing ===
    max_workers: int = 4           # Parallel workers for URL processing
    chunk_size: int = 10000        # Batch size for large operations

    # === Column Mappings (customize if your data has different column names) ===
    profit_column: str = "conversionsValue"  # Column representing revenue/profit
    weight_column: str = "conversions"       # Column for allocation weights

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# === Quick Access Presets ===

def default_config() -> PipelineConfig:
    """Get default configuration."""
    return PipelineConfig()


def local_config() -> PipelineConfig:
    """Configuration for local ClickHouse (no AWS)."""
    return PipelineConfig(
        use_aws_secrets=False,
        clickhouse_host="localhost",
        clickhouse_port=8123,
    )


def staging_config() -> PipelineConfig:
    """Configuration for staging environment."""
    return PipelineConfig(
        aws_profile="staging",
    )
