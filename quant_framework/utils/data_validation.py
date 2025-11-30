"""
Data validation utilities for the quantitative trading framework.

This module provides tools for validating market data, ensuring data quality,
and handling data anomalies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


@dataclass
class DataQualityReport:
    """Container for data quality report."""
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    date_range: Tuple[datetime, datetime]
    validation_result: ValidationResult


class DataValidator:
    """Validator for market data."""
    
    def __init__(self, 
                 price_change_threshold: float = 0.2,
                 volume_change_threshold: float = 5.0,
                 min_price: float = 0.01,
                 max_price_change_pct: float = 0.5):
        """
        Initialize data validator.
        
        Args:
            price_change_threshold: Threshold for significant price changes
            volume_change_threshold: Threshold for significant volume changes
            min_price: Minimum valid price
            max_price_change_pct: Maximum allowed price change percentage
        """
        self.price_change_threshold = price_change_threshold
        self.volume_change_threshold = volume_change_threshold
        self.min_price = min_price
        self.max_price_change_pct = max_price_change_pct
    
    def validate_ohlcv_data(self, data: pd.DataFrame, 
                            symbol: str = "") -> ValidationResult:
        """
        Validate OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            symbol: Symbol name for error messages
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings_list = []
        stats = {}
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append(f"Data is empty for symbol {symbol}")
            return ValidationResult(False, errors, warnings_list, stats)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return ValidationResult(False, errors, warnings_list, stats)
        
        # Check data types
        for col in required_columns:
            if not np.issubdtype(data[col].dtype, np.number):
                errors.append(f"Column {col} is not numeric")
        
        # Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            if (data[col] < 0).any():
                errors.append(f"Negative values found in {col} column")
        
        # Check for zero volume
        zero_volume_count = (data['volume'] == 0).sum()
        if zero_volume_count > 0:
            warnings_list.append(f"Found {zero_volume_count} records with zero volume")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            errors.append(f"Found {invalid_count} records with invalid OHLC relationships")
        
        # Check for extreme price changes
        if len(data) > 1:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > self.max_price_change_pct
            
            if extreme_changes.any():
                extreme_count = extreme_changes.sum()
                warnings_list.append(f"Found {extreme_count} records with extreme price changes (> {self.max_price_change_pct:.0%})")
        
        # Check for extreme volume changes
        if len(data) > 1:
            volume_changes = data['volume'].pct_change().abs()
            extreme_volume_changes = volume_changes > self.volume_change_threshold
            
            if extreme_volume_changes.any():
                extreme_count = extreme_volume_changes.sum()
                warnings_list.append(f"Found {extreme_count} records with extreme volume changes (> {self.volume_change_threshold:.0%})")
        
        # Calculate statistics
        stats = {
            'total_records': len(data),
            'date_range': (data.index.min(), data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
            'price_range': {
                'min': data[['open', 'high', 'low', 'close']].min().min(),
                'max': data[['open', 'high', 'low', 'close']].max().max()
            },
            'volume_stats': {
                'mean': data['volume'].mean(),
                'median': data['volume'].median(),
                'max': data['volume'].max()
            },
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_records': data.index.duplicated().sum()
        }
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings_list, stats)
    
    def validate_tick_data(self, data: pd.DataFrame, 
                          symbol: str = "") -> ValidationResult:
        """
        Validate tick data.
        
        Args:
            data: DataFrame with tick data columns
            symbol: Symbol name for error messages
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings_list = []
        stats = {}
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append(f"Data is empty for symbol {symbol}")
            return ValidationResult(False, errors, warnings_list, stats)
        
        # Check required columns
        required_columns = ['price', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return ValidationResult(False, errors, warnings_list, stats)
        
        # Check data types
        for col in required_columns:
            if not np.issubdtype(data[col].dtype, np.number):
                errors.append(f"Column {col} is not numeric")
        
        # Check for negative prices
        if (data['price'] < 0).any():
            errors.append("Negative values found in price column")
        
        # Check for zero volume
        zero_volume_count = (data['volume'] == 0).sum()
        if zero_volume_count > 0:
            warnings_list.append(f"Found {zero_volume_count} records with zero volume")
        
        # Check for extreme price changes
        if len(data) > 1:
            price_changes = data['price'].pct_change().abs()
            extreme_changes = price_changes > self.max_price_change_pct
            
            if extreme_changes.any():
                extreme_count = extreme_changes.sum()
                warnings_list.append(f"Found {extreme_count} records with extreme price changes (> {self.max_price_change_pct:.0%})")
        
        # Calculate statistics
        stats = {
            'total_records': len(data),
            'date_range': (data.index.min(), data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
            'price_stats': {
                'min': data['price'].min(),
                'max': data['price'].max(),
                'mean': data['price'].mean(),
                'median': data['price'].median()
            },
            'volume_stats': {
                'mean': data['volume'].mean(),
                'median': data['volume'].median(),
                'max': data['volume'].max()
            },
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_records': data.index.duplicated().sum()
        }
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings_list, stats)
    
    def validate_time_series(self, data: pd.Series, 
                           name: str = "") -> ValidationResult:
        """
        Validate time series data.
        
        Args:
            data: Series with time series data
            name: Series name for error messages
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings_list = []
        stats = {}
        
        # Check if Series is empty
        if data.empty:
            errors.append(f"Series is empty for {name}")
            return ValidationResult(False, errors, warnings_list, stats)
        
        # Check for NaN values
        nan_count = data.isnull().sum()
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values in series {name}")
        
        # Check for infinite values
        inf_count = np.isinf(data).sum()
        if inf_count > 0:
            errors.append(f"Found {inf_count} infinite values in series {name}")
        
        # Calculate statistics
        stats = {
            'total_records': len(data),
            'date_range': (data.index.min(), data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
            'stats': {
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std()
            },
            'missing_values': nan_count,
            'duplicate_records': data.index.duplicated().sum()
        }
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings_list, stats)
    
    def detect_outliers(self, data: pd.Series, method: str = "iqr", 
                       threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a data series.
        
        Args:
            data: Series with data
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
            
        elif method == "modified_zscore":
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def generate_data_quality_report(self, data: pd.DataFrame, 
                                   symbol: str = "") -> DataQualityReport:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data: DataFrame with market data
            symbol: Symbol name for the report
            
        Returns:
            DataQualityReport with detailed quality information
        """
        # Validate data
        validation_result = self.validate_ohlcv_data(data, symbol)
        
        # Count missing values
        missing_values = data.isnull().sum().to_dict()
        
        # Count duplicate records
        duplicate_records = data.index.duplicated().sum()
        
        # Detect outliers for each numeric column
        outliers = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            outlier_series = self.detect_outliers(data[col].dropna())
            outliers[col] = outlier_series.sum()
        
        # Get data types
        data_types = data.dtypes.astype(str).to_dict()
        
        # Get date range
        date_range = (None, None)
        if isinstance(data.index, pd.DatetimeIndex):
            date_range = (data.index.min(), data.index.max())
        
        return DataQualityReport(
            total_records=len(data),
            missing_values=missing_values,
            duplicate_records=duplicate_records,
            outliers=outliers,
            data_types=data_types,
            date_range=date_range,
            validation_result=validation_result
        )


class DataCleaner:
    """Cleaner for market data."""
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """
        Handle missing values in market data.
        
        Args:
            data: DataFrame with market data
            method: Method for handling missing values ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        data_copy = data.copy()
        
        if method == "forward_fill":
            data_copy = data_copy.fillna(method='ffill')
        elif method == "backward_fill":
            data_copy = data_copy.fillna(method='bfill')
        elif method == "interpolate":
            data_copy = data_copy.interpolate()
        elif method == "drop":
            data_copy = data_copy.dropna()
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
        
        return data_copy
    
    @staticmethod
    def handle_outliers(data: pd.DataFrame, method: str = "clip", 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle outliers in market data.
        
        Args:
            data: DataFrame with market data
            method: Method for handling outliers ('clip', 'remove', 'transform')
            columns: List of columns to process (all numeric columns if None)
            
        Returns:
            DataFrame with outliers handled
        """
        data_copy = data.copy()
        
        if columns is None:
            columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in data_copy.columns:
                continue
                
            if method == "clip":
                Q1 = data_copy[col].quantile(0.25)
                Q3 = data_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data_copy[col] = data_copy[col].clip(lower_bound, upper_bound)
                
            elif method == "remove":
                Q1 = data_copy[col].quantile(0.25)
                Q3 = data_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data_copy = data_copy[(data_copy[col] >= lower_bound) & (data_copy[col] <= upper_bound)]
                
            elif method == "transform":
                # Log transform for positive values
                if (data_copy[col] > 0).all():
                    data_copy[col] = np.log(data_copy[col])
                else:
                    # Box-Cox transform would be better, but requires scipy
                    # Using a simple winsorization for now
                    Q1 = data_copy[col].quantile(0.05)
                    Q3 = data_copy[col].quantile(0.95)
                    
                    data_copy[col] = data_copy[col].clip(Q1, Q3)
            
            else:
                raise ValueError(f"Unknown outlier handling method: {method}")
        
        return data_copy
    
    @staticmethod
    def fix_ohlc_relationships(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fix OHLC relationships in market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with fixed OHLC relationships
        """
        data_copy = data.copy()
        
        # Ensure high is the maximum of OHLC
        data_copy['high'] = data_copy[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Ensure low is the minimum of OHLC
        data_copy['low'] = data_copy[['open', 'high', 'low', 'close']].min(axis=1)
        
        return data_copy
    
    @staticmethod
    def remove_duplicates(data: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
        """
        Remove duplicate records from market data.
        
        Args:
            data: DataFrame with market data
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        return data.drop_duplicates(keep=keep)
    
    @staticmethod
    def align_time_series(data: pd.DataFrame, frequency: str = "1D", 
                         method: str = "forward_fill") -> pd.DataFrame:
        """
        Align time series to a regular frequency.
        
        Args:
            data: DataFrame with time series data
            frequency: Target frequency ('1D', '1H', '5T', etc.)
            method: Method for filling gaps ('forward_fill', 'interpolate', 'drop')
            
        Returns:
            DataFrame with aligned time series
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn("Data index is not DatetimeIndex, returning original data")
            return data
        
        # Create a complete date range
        date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq=frequency)
        
        # Reindex data to the complete date range
        aligned_data = data.reindex(date_range)
        
        # Handle missing values based on method
        if method == "forward_fill":
            aligned_data = aligned_data.fillna(method='ffill')
        elif method == "interpolate":
            aligned_data = aligned_data.interpolate()
        elif method == "drop":
            aligned_data = aligned_data.dropna()
        else:
            raise ValueError(f"Unknown gap filling method: {method}")
        
        return aligned_data


class DataProfiler:
    """Profiler for market data analysis."""
    
    @staticmethod
    def profile_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with data profile information
        """
        profile = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'date_range': None,
            'time_interval': None
        }
        
        # Add date range and time interval for time series data
        if isinstance(data.index, pd.DatetimeIndex):
            profile['date_range'] = (data.index.min(), data.index.max())
            
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                profile['time_interval'] = time_diffs.mode().iloc[0] if not time_diffs.empty else None
        
        # Add statistics for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            profile['statistics'] = data[numeric_columns].describe().to_dict()
        
        # Add missing value information
        profile['missing_values'] = data.isnull().sum().to_dict()
        profile['missing_percentage'] = (data.isnull().sum() / len(data) * 100).to_dict()
        
        return profile
    
    @staticmethod
    def analyze_price_patterns(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price patterns in market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with price pattern analysis
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return {'error': 'Required OHLC columns not found'}
        
        analysis = {}
        
        # Price change statistics
        analysis['price_changes'] = {
            'daily_returns': data['close'].pct_change().describe().to_dict(),
            'intraday_changes': ((data['close'] - data['open']) / data['open']).describe().to_dict(),
            'high_low_range': ((data['high'] - data['low']) / data['open']).describe().to_dict()
        }
        
        # Gap analysis
        if len(data) > 1:
            gaps = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            analysis['gaps'] = {
                'gap_up_count': (gaps > 0).sum(),
                'gap_down_count': (gaps < 0).sum(),
                'gap_stats': gaps.describe().to_dict()
            }
        
        # Candlestick patterns
        body_size = np.abs(data['close'] - data['open'])
        upper_shadow = data['high'] - np.maximum(data['close'], data['open'])
        lower_shadow = np.minimum(data['close'], data['open']) - data['low']
        
        analysis['candlestick_patterns'] = {
            'doji_count': (body_size < 0.01 * (data['high'] - data['low'])).sum(),
            'hammer_count': ((lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * body_size)).sum(),
            'shooting_star_count': ((upper_shadow > 2 * body_size) & (lower_shadow < 0.1 * body_size)).sum(),
            'body_size_stats': body_size.describe().to_dict()
        }
        
        return analysis
    
    @staticmethod
    def analyze_volume_patterns(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume patterns in market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume pattern analysis
        """
        if 'volume' not in data.columns:
            return {'error': 'Volume column not found'}
        
        analysis = {}
        
        # Volume statistics
        analysis['volume_stats'] = data['volume'].describe().to_dict()
        
        # Volume change statistics
        if len(data) > 1:
            volume_changes = data['volume'].pct_change()
            analysis['volume_changes'] = volume_changes.describe().to_dict()
            
            # High volume days (top 10%)
            high_volume_threshold = data['volume'].quantile(0.9)
            analysis['high_volume_days'] = {
                'count': (data['volume'] > high_volume_threshold).sum(),
                'threshold': high_volume_threshold
            }
        
        # Volume-price relationship
        if len(data) > 1:
            price_changes = data['close'].pct_change().dropna()
            volume_changes = data['volume'].pct_change().dropna()
            
            # Align the data
            common_index = price_changes.index.intersection(volume_changes.index)
            price_changes = price_changes.loc[common_index]
            volume_changes = volume_changes.loc[common_index]
            
            if len(common_index) > 0:
                correlation = price_changes.corr(volume_changes)
                analysis['volume_price_correlation'] = correlation
        
        return analysis