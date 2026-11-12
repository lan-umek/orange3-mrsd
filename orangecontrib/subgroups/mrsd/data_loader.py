"""
Data loading utilities for multi-target subgroup discovery.
Supports CSV, XLSX, and TXT formats with flexible variable definitions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

from .common import FeatureSpace, _is_categorical_dtype


@dataclass
class Dataset:
    """
    Container for dataset with separated independent (X) and dependent (Y) variables.
    
    Attributes:
        data: Full DataFrame with all variables
        X: DataFrame with independent variables only
        Y: DataFrame with dependent variables only
        X_columns: List of independent variable names
        Y_columns: List of dependent variable names
        unused_columns: Variables not assigned to X or Y
        n_instances: Number of data instances (rows)
    """
    data: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    X_columns: List[str]
    Y_columns: List[str]
    unused_columns: List[str] = field(default_factory=list)
    
    @property
    def n_instances(self) -> int:
        """Number of instances in the dataset."""
        return len(self.data)
    
    @property
    def n_X_variables(self) -> int:
        """Number of independent variables."""
        return len(self.X_columns)
    
    @property
    def n_Y_variables(self) -> int:
        """Number of dependent variables."""
        return len(self.Y_columns)
    
    def get_instance_indices(self) -> Set[int]:
        """Return set of all instance indices."""
        return set(range(self.n_instances))
    
    def is_X_categorical(self) -> bool:
        """True if ALL X variables are categorical (by dtype)."""
        return len(self.X_columns) > 0 and all(_is_categorical_dtype(self.X[col]) for col in self.X_columns)
    
    def is_Y_categorical(self) -> bool:
        """True if ALL Y variables are categorical (by dtype)."""
        return len(self.Y_columns) > 0 and all(_is_categorical_dtype(self.Y[col]) for col in self.Y_columns)

    def categorical_X_columns(self) -> List[str]:
        return [c for c in self.X_columns if _is_categorical_dtype(self.X[c])]

    def categorical_Y_columns(self) -> List[str]:
        return [c for c in self.Y_columns if _is_categorical_dtype(self.Y[c])]

    @property
    def X_space(self) -> FeatureSpace:
        """Typed numeric view of X (cached)."""
        if getattr(self, "_X_space", None) is None:
            self._X_space = FeatureSpace.from_dataframe(self.X)
        return self._X_space

    @property
    def Y_space(self) -> FeatureSpace:
        """Typed numeric view of Y (cached)."""
        if getattr(self, "_Y_space", None) is None:
            self._Y_space = FeatureSpace.from_dataframe(self.Y)
        return self._Y_space
    
    def count_natural_partitions_X(self) -> int:
        """Count number of unique combinations in X (for natural partitioning)."""
        if self.n_X_variables == 0:
            return 0
        return self.X.drop_duplicates().shape[0]
    
    def count_natural_partitions_Y(self) -> int:
        """Count number of unique combinations in Y (for natural partitioning)."""
        if self.n_Y_variables == 0:
            return 0
        return self.Y.drop_duplicates().shape[0]
    
    def summary(self) -> str:
        """Return a summary of the dataset."""
        lines = [
            f"Dataset Summary:",
            f"  Total instances: {self.n_instances}",
            f"  Independent variables (X): {self.n_X_variables} - {self.X_columns}",
            f"  Dependent variables (Y): {self.n_Y_variables} - {self.Y_columns}",
        ]
        if self.unused_columns:
            lines.append(f"  Unused variables: {self.unused_columns}")
        
        lines.append(f"  X categorical: {self.is_X_categorical()}")
        lines.append(f"  Y categorical: {self.is_Y_categorical()}")
        
        if self.is_X_categorical():
            lines.append(f"  Natural partitions in X: {self.count_natural_partitions_X()}")
        if self.is_Y_categorical():
            lines.append(f"  Natural partitions in Y: {self.count_natural_partitions_Y()}")
            
        return "\n".join(lines)


class DataLoader:
    """
    Utility class for loading datasets from various file formats.
    """
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.txt', '.tsv'}
    
    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        X_columns: Optional[List[str]] = None,
        Y_columns: Optional[List[str]] = None,
        separator: str = ',',
        header: int = 0,
        na_values: Optional[List[str]] = None,
        drop_na: bool = True
    ) -> Dataset:
        """
        Load dataset from file and split into X and Y variables.
        
        Args:
            filepath: Path to the data file
            X_columns: List of independent variable names. If None, must be set later.
            Y_columns: List of dependent variable names. If None, must be set later.
            separator: Column separator for CSV/TXT files (default: ',')
            header: Row number for column names (default: 0)
            na_values: Additional strings to recognize as NA/NaN
            drop_na: Whether to drop rows with missing values (default: True)
            
        Returns:
            Dataset object with X and Y separated
            
        Raises:
            ValueError: If file format is not supported or columns not found
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        if suffix not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}. "
                           f"Supported: {cls.SUPPORTED_FORMATS}")
        
        # Load data based on format
        if suffix in {'.csv', '.txt', '.tsv'}:
            if suffix == '.tsv':
                separator = '\t'
            data = pd.read_csv(filepath, sep=separator, header=header, 
                              na_values=na_values)
        elif suffix in {'.xlsx', '.xls'}:
            data = pd.read_excel(filepath, header=header, na_values=na_values)
        
        # Handle missing values (only in the columns that are actually used)
        if drop_na:
            original_len = len(data)
            used = [c for c in (X_columns or []) + (Y_columns or []) if c in data.columns]
            data = data.dropna(subset=used) if used else data.dropna()
            if len(data) < original_len:
                print(f"Dropped {original_len - len(data)} rows with missing values in X/Y columns")
        
        # Reset index after potential row drops
        data = data.reset_index(drop=True)
        
        return cls._create_dataset(data, X_columns, Y_columns)
    
    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        X_columns: Optional[List[str]] = None,
        Y_columns: Optional[List[str]] = None
    ) -> Dataset:
        """
        Create Dataset from existing pandas DataFrame.
        
        Args:
            data: pandas DataFrame
            X_columns: List of independent variable names
            Y_columns: List of dependent variable names
            
        Returns:
            Dataset object
        """
        data = data.reset_index(drop=True)
        return cls._create_dataset(data.copy(), X_columns, Y_columns)
    
    @classmethod
    def _create_dataset(
        cls,
        data: pd.DataFrame,
        X_columns: Optional[List[str]],
        Y_columns: Optional[List[str]]
    ) -> Dataset:
        """
        Internal method to create Dataset object.
        """
        all_columns = set(data.columns)
        
        # Validate X columns
        if X_columns is None:
            X_columns = []
        else:
            missing_X = set(X_columns) - all_columns
            if missing_X:
                raise ValueError(f"X columns not found in data: {missing_X}")
        
        # Validate Y columns
        if Y_columns is None:
            Y_columns = []
        else:
            missing_Y = set(Y_columns) - all_columns
            if missing_Y:
                raise ValueError(f"Y columns not found in data: {missing_Y}")
        
        # Check for overlap
        overlap = set(X_columns) & set(Y_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both X and Y: {overlap}")
        if not X_columns or not Y_columns:
            raise ValueError("Both X_columns and Y_columns must be non-empty")
        used = list(X_columns) + list(Y_columns)
        if data[used].isna().any().any():
            raise ValueError("Missing values in X/Y columns; use DataLoader.load(drop_na=True) or impute first")
        
        # Identify unused columns
        used_columns = set(X_columns) | set(Y_columns)
        unused_columns = list(all_columns - used_columns)
        
        # Create X and Y DataFrames
        X = data[X_columns] if X_columns else pd.DataFrame(index=data.index)
        Y = data[Y_columns] if Y_columns else pd.DataFrame(index=data.index)
        
        return Dataset(
            data=data,
            X=X,
            Y=Y,
            X_columns=list(X_columns),
            Y_columns=list(Y_columns),
            unused_columns=unused_columns
        )
    
    @staticmethod
    def list_columns(filepath: Union[str, Path], separator: str = ',') -> List[str]:
        """
        List column names in a file without loading all data.
        
        Args:
            filepath: Path to the data file
            separator: Column separator for CSV/TXT files
            
        Returns:
            List of column names
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix in {'.csv', '.txt', '.tsv'}:
            if suffix == '.tsv':
                separator = '\t'
            df = pd.read_csv(filepath, sep=separator, nrows=0)
        elif suffix in {'.xlsx', '.xls'}:
            df = pd.read_excel(filepath, nrows=0)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        return list(df.columns)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'material': np.random.choice(['steel', 'iron', 'wood'], 100),
        'length': np.random.uniform(50, 500, 100),
        'lanes': np.random.choice([2, 4, 6], 100),
        'erected': np.random.choice(['modern', 'old', 'ancient'], 100),
        'condition': np.random.choice(['good', 'fair', 'poor'], 100),
        'load_capacity': np.random.uniform(10, 100, 100),
        'id': range(100)  # unused column
    })
    
    # Save to CSV for testing
    sample_data.to_csv('/tmp/test_bridges.csv', index=False)
    
    # Load with column specification
    dataset = DataLoader.load(
        '/tmp/test_bridges.csv',
        X_columns=['material', 'length', 'lanes'],
        Y_columns=['condition', 'load_capacity']
    )
    
    print(dataset.summary())
    print("\nFirst 5 rows of X:")
    print(dataset.X.head())
    print("\nFirst 5 rows of Y:")
    print(dataset.Y.head())
