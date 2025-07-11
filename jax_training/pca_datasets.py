import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
# from torchvision import datasets, transforms

from scipy.io import arff
def generate_dataset_semeion_pca_from_file(n_components=8, test_size=0.15, val_size=0.15, random_state=42, arff_file="phptd5jYj.arff"):
    """
    Loads the Semeion dataset from an ARFF file, applies PCA to reduce it to n_components,
    scales the PCA features to [0, 1], and splits the data into training, validation, and test sets.
    
    Parameters:
      n_components (int): Number of PCA components/features.
      test_size (float): Fraction of the full dataset to reserve for testing.
                           If set to 0, no test split is created.
      val_size (float): Fraction of the (train+validation) split to use as validation.
                        If set to 0, no validation split is created.
      random_state (int): Random seed for reproducibility.
      arff_file (str): Filename or path to the ARFF file.
      
    Returns:
      X_train (jnp.array): Training set features after PCA and scaling.
      y_train (jnp.array): Training set labels.
      X_val (jnp.array): Validation set features after PCA and scaling.
                        Returns an empty array if val_size==0.
      y_val (jnp.array): Validation set labels.
                        Returns an empty array if val_size==0.
      X_test (jnp.array): Test set features after PCA and scaling.
                        Returns an empty array if test_size==0.
      y_test (jnp.array): Test set labels.
                        Returns an empty array if test_size==0.
    """
    # Load ARFF file using scipy.io.arff
    data_arff, meta = arff.loadarff(arff_file)
    df = pd.DataFrame(data_arff)
    
    # The ARFF file contains 256 feature columns and one label column (named "Class").
    # In your printed output, the features appear as V1, V2, ..., V256 and the label column as "Class".
    feature_cols = [f"V{i}" for i in range(1, 257)]
    label_col = "Class"
    
    # Convert the features. (They should be numeric; if they are bytes, convert them to float.)
    data = df[feature_cols].values.astype(np.float32)
    
    # The Class column might be stored as bytes; convert it to integer labels.
    if df[label_col].dtype == object:
        labels = df[label_col].apply(lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x)).values.astype(np.int32)
    else:
        labels = df[label_col].values.astype(np.int32)
    
    # First split: separate out the test set if test_size > 0.
    if test_size > 0.0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_val, y_train_val = data, labels
        X_test = np.empty((0, data.shape[1]), dtype=np.float32)
        y_test = np.empty((0,), dtype=labels.dtype)
    
    # Second split: separate training and validation sets if val_size > 0.
    if val_size > 0.0:
        # When test_size > 0, X_train_val is 1 - test_size fraction of data;
        # the effective validation size is adjusted accordingly.
        test_size_val = val_size if test_size == 0.0 else val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_val, random_state=random_state, stratify=y_train_val
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, X_train_val.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=y_train_val.dtype)
    
    # Apply PCA on the training data only.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val) if X_val.shape[0] > 0 else np.empty((0, n_components), dtype=np.float32)
    X_test_pca = pca.transform(X_test) if X_test.shape[0] > 0 else np.empty((0, n_components), dtype=np.float32)
    
    # Scale the PCA features to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca) if X_val_pca.shape[0] > 0 else X_val_pca
    X_test_scaled = scaler.transform(X_test_pca) if X_test_pca.shape[0] > 0 else X_test_pca
    
    # Convert the numpy arrays to JAX arrays.
    return (
        jnp.array(X_train_scaled, dtype=jnp.float32),
        jnp.array(y_train, dtype=jnp.int32),
        jnp.array(X_val_scaled, dtype=jnp.float32),
        jnp.array(y_val, dtype=jnp.int32),
        jnp.array(X_test_scaled, dtype=jnp.float32),
        jnp.array(y_test, dtype=jnp.int32)
    )


def generate_dataset_semeion_pca(n_components=8, test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads the Semeion dataset from the UCI repository, applies PCA to reduce to n_components, 
    scales the PCA features to [0, 1], and splits the data into training, validation, and test sets.
    Returns the datasets as JAX arrays.

    Parameters:
      n_components (int): Number of PCA components/features.
      test_size (float): Fraction of the full dataset to reserve for testing.
                           If set to 0, no test split is created.
      val_size (float): Fraction of the (train+validation) split to use as validation.
                        If set to 0, no validation split is created.
      random_state (int): Random seed for reproducibility.

    Returns:
      X_train (jnp.array): Training set features after PCA and scaling.
      y_train (jnp.array): Training set labels.
      X_val (jnp.array): Validation set features after PCA and scaling.
                        Returns an empty array if val_size is 0.
      y_val (jnp.array): Validation set labels.
                        Returns an empty array if val_size is 0.
      X_test (jnp.array): Test set features after PCA and scaling.
                        Returns an empty array if test_size is 0.
      y_test (jnp.array): Test set labels.
                        Returns an empty array if test_size is 0.
    """
    # Download the Semeion dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
    df = pd.read_csv(url, sep='\s+', header=None)

    # The dataset has 256 feature columns (pixel values) and 10 columns for one-hot encoded labels
    data = df.iloc[:, :256].values
    labels_onehot = df.iloc[:, 256:].values

    # Convert one-hot encoded labels to single integer labels
    labels = np.argmax(labels_onehot, axis=1)

    # First split: (Train+Validation) and Test.
    if test_size > 0.0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_val, y_train_val = data, labels
        X_test = np.empty((0, data.shape[1]))
        y_test = np.empty((0,), dtype=labels.dtype)

    # Second split: Training and Validation.
    if val_size > 0.0:
        # If test_size==0, the fraction for validation is just val_size.
        test_size_val = val_size if test_size == 0.0 else val_size/(1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_val, random_state=random_state, stratify=y_train_val
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, X_train_val.shape[1]))
        y_val = np.empty((0,), dtype=y_train_val.dtype)

    # Apply PCA on the training data only.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    # Only apply PCA transform on test set if it has samples.
    if X_test.shape[0] > 0:
        X_test_pca = pca.transform(X_test)
    else:
        X_test_pca = np.empty((0, n_components))

    # Scale the PCA features to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca)
    # Only transform test set if non-empty.
    if X_test_pca.shape[0] > 0:
        X_test_scaled = scaler.transform(X_test_pca)
    else:
        X_test_scaled = np.empty((0, n_components))

    # Convert the numpy arrays to JAX arrays.
    return (jnp.array(X_train_scaled, jnp.float32), jnp.array(y_train, jnp.int32),
            jnp.array(X_val_scaled, jnp.float32), jnp.array(y_val, jnp.int32),
            jnp.array(X_test_scaled, jnp.float32), jnp.array(y_test, jnp.int32))


def generate_dataset_fashion_mnist_pca(n_components=8, test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads Fashion-MNIST, applies PCA to reduce to n_components, scales the PCA features to [0, 1],
    and splits the data into training, validation, and test sets. Supports test_size or val_size = 0.
    
    Parameters:
      n_components (int): Number of PCA components/features.
      test_size (float): Fraction of the full dataset to reserve for testing. If 0, no test set is created.
      val_size (float): Fraction of the (train+val) split to use as validation. If 0, no validation set is created.
      random_state (int): Random seed for reproducibility.
      
    Returns:
      X_train (jnp.array): Training set features after PCA and scaling.
      y_train (jnp.array): Training set labels.
      X_val (jnp.array): Validation set features after PCA and scaling (empty if val_size==0).
      y_val (jnp.array): Validation set labels (empty if val_size==0).
      X_test (jnp.array): Test set features after PCA and scaling (empty if test_size==0).
      y_test (jnp.array): Test set labels (empty if test_size==0).
    """
    import numpy as np
    from torchvision import datasets, transforms
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    import jax.numpy as jnp

    # Define a transform to convert images to flattened tensors.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor (values in [0, 1])
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 image to a vector
    ])
    
    # Load Fashion-MNIST datasets.
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract data and labels as numpy arrays.
    X_train_full = np.array([np.array(img) for img, _ in fashion_train])
    y_train_full = np.array([label for _, label in fashion_train])
    X_test_full = np.array([np.array(img) for img, _ in fashion_test])
    y_test_full = np.array([label for _, label in fashion_test])
    
    # Combine train and test sets for a full dataset.
    dataset = np.concatenate([X_train_full, X_test_full], axis=0)
    labels = np.concatenate([y_train_full, y_test_full], axis=0)
    
    # First split: Separate test set if test_size > 0.
    if test_size > 0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            dataset, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_val = dataset
        y_train_val = labels
        X_test = np.empty((0, dataset.shape[1]))
        y_test = np.empty((0,), dtype=labels.dtype)
    
    # Second split: Create validation set if val_size > 0.
    if val_size > 0:
        test_size_for_val = val_size / (1 - test_size) if test_size > 0 else val_size
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_for_val, random_state=random_state, stratify=y_train_val
        )
    else:
        X_train = X_train_val
        X_val = np.empty((0, X_train_val.shape[1]))
        y_val = np.empty((0,), dtype=y_train_val.dtype)
    
    # Apply PCA on the training data only.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val) if X_val.shape[0] > 0 else X_val
    X_test_pca = pca.transform(X_test) if X_test.shape[0] > 0 else X_test
    
    # Scale the PCA features to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca) if X_val_pca.shape[0] > 0 else X_val_pca
    X_test_scaled = scaler.transform(X_test_pca) if X_test_pca.shape[0] > 0 else X_test_pca
    
    # Convert the numpy arrays to JAX arrays.
    return (jnp.array(X_train_scaled, dtype=jnp.float32), jnp.array(y_train, dtype=jnp.int32),
            jnp.array(X_val_scaled, dtype=jnp.float32), jnp.array(y_val, dtype=jnp.int32),
            jnp.array(X_test_scaled, dtype=jnp.float32), jnp.array(y_test, dtype=jnp.int32))



from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import numpy as np

def generate_dataset_diabetes_pca(
        n_components: int = 8,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ):
    """
    Loads the Diabetes dataset, applies PCA to reduce it to n_components,
    scales *both* the PCA features **and the targets** to [0, 1], and splits
    the data into training, validation, and test sets.
    """
    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    diabetes = load_diabetes()
    data, targets = diabetes.data.astype(np.float32), diabetes.target.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. First split: (train+val) vs test
    # ------------------------------------------------------------------
    if test_size > 0.0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, targets,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        X_train_val, y_train_val = data, targets
        X_test  = np.empty((0, data.shape[1]), dtype=np.float32)
        y_test  = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Second split: train vs val
    # ------------------------------------------------------------------
    if val_size > 0.0:
        test_size_val = val_size if test_size == 0.0 else val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=test_size_val,
            random_state=random_state,
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, X_train_val.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Fit PCA on training data only
    # ------------------------------------------------------------------
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)   if X_val.shape[0]  > 0 else np.empty((0, n_components), dtype=np.float32)
    X_test_pca  = pca.transform(X_test)  if X_test.shape[0] > 0 else np.empty((0, n_components), dtype=np.float32)

    # ------------------------------------------------------------------
    # 5a. Scale PCA features to [0, 1]
    # ------------------------------------------------------------------
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = x_scaler.fit_transform(X_train_pca)
    X_val_scaled   = x_scaler.transform(X_val_pca)  if X_val_pca.shape[0]  > 0 else X_val_pca
    X_test_scaled  = x_scaler.transform(X_test_pca) if X_test_pca.shape[0] > 0 else X_test_pca

    # ------------------------------------------------------------------
    # 5b. **Scale targets to [0, 1]**
    # ------------------------------------------------------------------
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled   = (
        y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        if y_val.shape[0] > 0 else y_val
    )
    y_test_scaled  = (
        y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        if y_test.shape[0] > 0 else y_test
    )

    # ------------------------------------------------------------------
    # 6. Convert to JAX arrays and return
    # ------------------------------------------------------------------
    return (
        jnp.array(X_train_scaled, dtype=jnp.float32),
        jnp.array(y_train_scaled, dtype=jnp.float32),
        jnp.array(X_val_scaled,   dtype=jnp.float32),
        jnp.array(y_val_scaled,   dtype=jnp.float32),
        jnp.array(X_test_scaled,  dtype=jnp.float32),
        jnp.array(y_test_scaled,  dtype=jnp.float32),
    )


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp

def generate_dataset_wine_pca(
        n_components: int = 8,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        red: bool = True,
    ):
    """
    Loads the Wine-Quality dataset (red by default, white if ``red=False``),
    applies PCA to reduce it to ``n_components``, scales *both* the PCA
    features **and the targets**, and returns JAX arrays split into
    train / val / test just like ``generate_dataset_diabetes_pca``.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep (≤ 11).
    test_size : float
        Fraction of the data reserved for the test set.
    val_size : float
        Fraction reserved for the validation set (taken from the
        remaining train+val split).
    random_state : int
        Reproducibility seed for both splits and PCA.
    red : bool
        If True, use the *red* wine subset (1 599 rows);
        else use the *white* subset (4 898 rows).

    Returns
    -------
    tuple of jnp.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test), each
        already scaled and cast to ``jnp.float32``.
    """
    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    base_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-{}.csv"
    )
    csv_url = base_url.format("red" if red else "white")
    df = pd.read_csv(csv_url, sep=";").astype(np.float32)

    data    = df.drop(columns=["quality"]).values           # shape (N, 11)
    targets = df["quality"].values                          # shape (N,)

    # Safety check
    if n_components > data.shape[1]:
        raise ValueError(
            f"n_components={n_components} is greater than the "
            f"number of input features ({data.shape[1]})."
        )

    # ------------------------------------------------------------------
    # 2. First split: (train+val) vs test
    # ------------------------------------------------------------------
    if test_size > 0.0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, targets, test_size=test_size, random_state=random_state
        )
    else:
        X_train_val, y_train_val = data, targets
        X_test  = np.empty((0, data.shape[1]), dtype=np.float32)
        y_test  = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Second split: train vs val
    # ------------------------------------------------------------------
    if val_size > 0.0:
        test_size_val = val_size if test_size == 0.0 else val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=test_size_val,
            random_state=random_state,
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, X_train_val.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Fit PCA on training data only
    # ------------------------------------------------------------------
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)   if X_val.size  else np.empty((0, n_components), dtype=np.float32)
    X_test_pca  = pca.transform(X_test)  if X_test.size else np.empty((0, n_components), dtype=np.float32)

    # ------------------------------------------------------------------
    # 5a. Scale PCA features to [0, 1]
    # ------------------------------------------------------------------
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = x_scaler.fit_transform(X_train_pca)
    X_val_scaled   = x_scaler.transform(X_val_pca)  if X_val_pca.size  else X_val_pca
    X_test_scaled  = x_scaler.transform(X_test_pca) if X_test_pca.size else X_test_pca

    # ------------------------------------------------------------------
    # 5b. Scale targets to [-1, 1]         (matches the Diabetes helper)
    # ------------------------------------------------------------------
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()   if y_val.size  else y_val
    y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()  if y_test.size else y_test

    # ------------------------------------------------------------------
    # 6. Convert to JAX arrays and return
    # ------------------------------------------------------------------
    return (
        jnp.array(X_train_scaled, dtype=jnp.float32),
        jnp.array(y_train_scaled, dtype=jnp.float32),
        jnp.array(X_val_scaled,   dtype=jnp.float32),
        jnp.array(y_val_scaled,   dtype=jnp.float32),
        jnp.array(X_test_scaled,  dtype=jnp.float32),
        jnp.array(y_test_scaled,  dtype=jnp.float32),
    )



import io, urllib.request, urllib.error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import jax.numpy as jnp


def generate_dataset_concrete_pca(
        n_components: int = 8,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        csv_url: str | None = None,
):
    """
    Concrete-Compressive-Strength helper (robust version).
    Tries UCI → GitHub CSV → user-supplied URL, then fails gracefully.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test) as jnp.float32,
    with PCA features scaled to [0, 1] and targets to [-1, 1].
    """
    # ------------------------------------------------------------------
    # 0. Decide where to download from
    # ------------------------------------------------------------------
    fallback_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "concrete/compressive/Concrete_Data.xls",               # XLS (official)  :contentReference[oaicite:2]{index=2}
        "https://raw.githubusercontent.com/stedy/"
        "Machine-Learning-with-R-datasets/master/concrete.csv",  # CSV mirror     :contentReference[oaicite:3]{index=3}
    ]
    if csv_url:
        # put user-supplied URL at the front of the queue
        fallback_urls.insert(0, csv_url)

    errors: list[str] = []
    df = None
    for url in fallback_urls:
        try:
            with urllib.request.urlopen(url) as resp:
                raw = resp.read()
            if url.lower().endswith((".xls", ".xlsx")):
                # need xlrd for .xls; openpyxl for .xlsx
                df = pd.read_excel(io.BytesIO(raw)).astype(np.float32)
            else:
                df = pd.read_csv(io.BytesIO(raw)).astype(np.float32)
            break  # success
        except Exception as e:
            errors.append(f"{url}: {e}")

    if df is None:
        joined = "\n  • ".join(errors)
        raise RuntimeError(
            "Couldn’t download the Concrete data from any source.\n"
            "Tried:\n  • " + joined +
            "\n\nFixes:\n"
            " ▸ Ensure the machine has internet access (https).\n"
            " ▸ For the XLS route, `pip install xlrd` (<= 2.0.x) is required.\n"
            " ▸ Or pass your own `csv_url=` pointing to a local/remote copy."
        )

    # ------------------------------------------------------------------
    # 1. Harmonise column names
    # ------------------------------------------------------------------
    # Excel uses the very long name, CSV uses 'strength'
    target_col = next(c for c in df.columns
                      if "strength" in c.lower())
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    if n_components > X.shape[1]:
        raise ValueError(
            f"n_components ({n_components}) > available features ({X.shape[1]})"
        )

    # ------------------------------------------------------------------
    # 2. Splits
    # ------------------------------------------------------------------
    if test_size > 0.0:
        X_trval, X_test, y_trval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        X_trval, y_trval = X, y
        X_test  = np.empty((0, X.shape[1]), dtype=np.float32)
        y_test  = np.empty((0,),            dtype=np.float32)

    if val_size > 0.0:
        val_ratio = val_size if test_size == 0 else val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trval, y_trval,
            test_size=val_ratio, random_state=random_state)
    else:
        X_train, y_train = X_trval, y_trval
        X_val  = np.empty((0, X.shape[1]), dtype=np.float32)
        y_val  = np.empty((0,),            dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. PCA on training data
    # ------------------------------------------------------------------
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)   if X_val.size  else X_val
    X_test_pca  = pca.transform(X_test)  if X_test.size else X_test

    # ------------------------------------------------------------------
    # 4. Scale features & target
    # ------------------------------------------------------------------
    xs = MinMaxScaler((0, 1)).fit(X_train_pca)
    ys = MinMaxScaler((-1, 1)).fit(y_train.reshape(-1, 1))

    X_train_s = xs.transform(X_train_pca)
    X_val_s   = xs.transform(X_val_pca)   if X_val_pca.size  else X_val_pca
    X_test_s  = xs.transform(X_test_pca)  if X_test_pca.size else X_test_pca

    y_train_s = ys.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = ys.transform(y_val.reshape(-1, 1)).ravel()   if y_val.size  else y_val
    y_test_s  = ys.transform(y_test.reshape(-1, 1)).ravel()  if y_test.size else y_test

    # ------------------------------------------------------------------
    # 5. Return JAX arrays
    # ------------------------------------------------------------------
    return tuple(map(lambda a: jnp.asarray(a, dtype=jnp.float32),
                     (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)))



import io, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import jax.numpy as jnp


def generate_dataset_energy_pca(
        n_components: int = 8,
        *,
        target: str = "heating",      # "heating" -> Y1,  "cooling" -> Y2
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        xls_url: str | None = None,
):
    """
    Energy-Efficiency (ENB2012) helper returning
    (X_train, y_train, X_val, y_val, X_test, y_test) as JAX arrays.
    PCA features are scaled to [0,1]; target to [-1,1].
    """
    # --------------------------------------------------------------- #
    # 1. Download & load                                              #
    # --------------------------------------------------------------- #
    default_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00242/ENB2012_data.xlsx"
    )
    with urllib.request.urlopen(xls_url or default_url) as resp:
        raw = resp.read()
    df = pd.read_excel(io.BytesIO(raw)).astype(np.float32)

    # --------------------------------------------------------------- #
    # 2. Resolve feature & target columns (supports both header styles)
    # --------------------------------------------------------------- #
    cols = df.columns.str.strip()          # remove stray spaces
    if {"X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"}.issubset(cols):
        feature_cols = [f"X{i}" for i in range(1, 9)]
        target_col = "Y1" if target.lower().startswith("h") else "Y2"
    else:  # fall back to descriptive labels
        feature_cols = [
            "Relative Compactness", "Surface Area", "Wall Area", "Roof Area",
            "Overall Height", "Orientation", "Glazing Area",
            "Glazing Area Distribution",
        ]
        target_col = "Heating Load" if target.lower().startswith("h") else "Cooling Load"

    X = df[feature_cols].values
    y = df[target_col].values

    if n_components > X.shape[1]:
        raise ValueError(f"`n_components` ({n_components}) > {X.shape[1]} features")

    # --------------------------------------------------------------- #
    # 3. Split: (train+val) vs test, then train vs val                #
    # --------------------------------------------------------------- #
    if test_size:
        X_trval, X_test, y_trval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        X_trval, y_trval = X, y
        X_test  = np.empty((0, X.shape[1]), np.float32)
        y_test  = np.empty((0,), np.float32)

    if val_size:
        val_ratio = val_size if test_size == 0 else val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trval, y_trval, test_size=val_ratio, random_state=random_state)
    else:
        X_train, y_train = X_trval, y_trval
        X_val = np.empty((0, X.shape[1]), np.float32)
        y_val = np.empty((0,), np.float32)

    # --------------------------------------------------------------- #
    # 4. PCA fit on training only                                     #
    # --------------------------------------------------------------- #
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)   if X_val.size  else X_val
    X_test_pca  = pca.transform(X_test)  if X_test.size else X_test

    # --------------------------------------------------------------- #
    # 5. Scale features & target                                      #
    # --------------------------------------------------------------- #
    x_scaler = MinMaxScaler((0, 1)).fit(X_train_pca)
    y_scaler = MinMaxScaler((-1, 1)).fit(y_train.reshape(-1, 1))

    X_train_s = x_scaler.transform(X_train_pca)
    X_val_s   = x_scaler.transform(X_val_pca)   if X_val_pca.size  else X_val_pca
    X_test_s  = x_scaler.transform(X_test_pca)  if X_test_pca.size else X_test_pca

    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()   if y_val.size  else y_val
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()  if y_test.size else y_test

    # --------------------------------------------------------------- #
    # 6. Return JAX arrays                                            #
    # --------------------------------------------------------------- #
    to_jax = lambda a: jnp.asarray(a, dtype=jnp.float32)
    return tuple(map(to_jax,
                     (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)))




import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp


def generate_dataset_synthetic(
        n_components: int = 8,          # ← now equals n_features
        test_size: float = 0.20,
        val_size: float = 0.20,
        random_state: int = 42,
        *,
        # --- extra knobs forwarded to make_regression ---------------
        n_samples: int = 1000,
        n_informative: int | None = None,
        noise: float = 10.0,
        bias: float = 0.0,
        n_targets: int = 1,
):
    """
    Create a synthetic regression problem with `sklearn.datasets.make_regression`
    and return (X_train, y_train, X_val, y_val, X_test, y_test) as JAX arrays.

    * `n_components` now controls the **number of generated features**.
    * Features are scaled to [0, 1]; targets to [-1, 1].
    * No PCA (hence the helper name without `_pca`).

    Parameters
    ----------
    n_components : int
        Number of predictors (`n_features`) to generate.
    test_size, val_size : float
        Fractions for the test and validation splits.  If both are non-zero,
        the validation fraction is taken from the remaining training portion.
    random_state : int
        Seed used for `make_regression` and the two splits.
    n_samples, n_informative, noise, bias, n_targets : *
        Passed straight to `make_regression`.

    Returns
    -------
    tuple[jnp.ndarray, ...]
        (X_train, y_train, X_val, y_val, X_test, y_test),
        already scaled and cast to `jnp.float32`.
    """
    # --------------------------------------------------------------- #
    # 1. Generate raw data                                            #
    # --------------------------------------------------------------- #
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_components,
        n_informative=(n_informative or n_components),
        noise=noise,
        bias=bias,
        n_targets=n_targets,
        random_state=random_state,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # --------------------------------------------------------------- #
    # 2. Train+val vs test split                                      #
    # --------------------------------------------------------------- #
    if test_size > 0.0:
        X_trval, X_test, y_trval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_trval, y_trval = X, y
        X_test  = np.empty((0, n_components), np.float32)
        y_test  = np.empty_like(y_trval[:0])

    # --------------------------------------------------------------- #
    # 3. Train vs val split                                           #
    # --------------------------------------------------------------- #
    if val_size > 0.0:
        val_ratio = val_size if test_size == 0.0 else val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trval, y_trval, test_size=val_ratio, random_state=random_state
        )
    else:
        X_train, y_train = X_trval, y_trval
        X_val = np.empty((0, n_components), np.float32)
        y_val = np.empty_like(y_trval[:0])

    # --------------------------------------------------------------- #
    # 4. Scale features to [0,1] and targets to [-1,1]                #
    # --------------------------------------------------------------- #
    x_scaler = MinMaxScaler((0, 1)).fit(X_train)
    y_scaler = MinMaxScaler((-1, 1)).fit(
        y_train.reshape(-1, n_targets)
    )

    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)   if X_val.size  else X_val
    X_test_s  = x_scaler.transform(X_test)  if X_test.size else X_test

    y_train_s = y_scaler.transform(y_train.reshape(-1, n_targets)).astype(np.float32)
    y_val_s   = (y_scaler.transform(y_val.reshape(-1, n_targets))
                 if y_val.size else y_val).astype(np.float32)
    y_test_s  = (y_scaler.transform(y_test.reshape(-1, n_targets))
                 if y_test.size else y_test).astype(np.float32)

    # flatten back when n_targets == 1
    if n_targets == 1:
        y_train_s = y_train_s.ravel()
        y_val_s   = y_val_s.ravel()
        y_test_s  = y_test_s.ravel()

    # --------------------------------------------------------------- #
    # 5. Return JAX arrays                                            #
    # --------------------------------------------------------------- #
    to_jax = lambda arr: jnp.asarray(arr, dtype=jnp.float32)
    return tuple(map(to_jax,
                     (X_train_s, y_train_s,
                      X_val_s,   y_val_s,
                      X_test_s,  y_test_s)))



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
import urllib.request
import io


def generate_dataset_airfoil(
        n_components: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        *,
        url: str | None = None,
):
    """
    Airfoil Self-Noise regression helper (NO PCA).

    • Downloads the 1 503-row UCI dataset
      (or a user-supplied `url` pointing to the same format).
    • Keeps the first `n_components` of the five available predictors.
    • Scales features to [0, 1] and the target to [-1, 1].
    • Splits into train / val / test and returns JAX float32 arrays.

    Returns
    -------
    tuple(jnp.ndarray, …)
        (X_train, y_train, X_val, y_val, X_test, y_test).

    Notes
    -----
    Original column order in the raw file:

        1. Frequency [Hz]
        2. Angle of attack [deg]
        3. Chord length [m]
        4. Free-stream velocity [m/s]
        5. Suction-side displacement thickness [m]
        6. Scaled sound-pressure level [dB]  ← target
    """
    # ------------------------------------------------------------------
    # 1. Download & load
    # ------------------------------------------------------------------
    default_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00291/airfoil_self_noise.dat"
    )
    with urllib.request.urlopen(url or default_url) as resp:
        raw = resp.read()

    df = pd.read_csv(io.BytesIO(raw),
                     delim_whitespace=True, header=None,
                     names=["frequency", "angle", "chord", "velocity",
                            "thickness", "sound_level"]).astype(np.float32)

    if not (1 <= n_components <= 5):
        raise ValueError("`n_components` must be 1–5 for Airfoil dataset.")

    X = df.iloc[:, :n_components].values              # predictors
    y = df["sound_level"].values                      # target (shape (1503,))

    # ------------------------------------------------------------------
    # 2. (train+val) vs test split
    # ------------------------------------------------------------------
    if test_size > 0:
        X_trval, X_test, y_trval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        X_trval, y_trval = X, y
        X_test  = np.empty((0, n_components), np.float32)
        y_test  = np.empty((0,),              np.float32)

    # ------------------------------------------------------------------
    # 3. train vs val split
    # ------------------------------------------------------------------
    if val_size > 0:
        val_ratio = val_size if test_size == 0 else val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trval, y_trval, test_size=val_ratio, random_state=random_state)
    else:
        X_train, y_train = X_trval, y_trval
        X_val = np.empty((0, n_components), np.float32)
        y_val = np.empty((0,),              np.float32)

    # ------------------------------------------------------------------
    # 4. Scale features → [0, 1], target → [-1, 1]
    # ------------------------------------------------------------------
    x_scaler = MinMaxScaler((0, 1)).fit(X_train)
    y_scaler = MinMaxScaler((-1, 1)).fit(y_train.reshape(-1, 1))

    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)   if X_val.size  else X_val
    X_test_s  = x_scaler.transform(X_test)  if X_test.size else X_test

    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = (y_scaler.transform(y_val.reshape(-1, 1)).ravel()
                 if y_val.size else y_val)
    y_test_s  = (y_scaler.transform(y_test.reshape(-1, 1)).ravel()
                 if y_test.size else y_test)

    # ------------------------------------------------------------------
    # 5. Return JAX arrays
    # ------------------------------------------------------------------
    to_jax = lambda a: jnp.asarray(a, dtype=jnp.float32)
    return tuple(map(to_jax,
                     (X_train_s, y_train_s,
                      X_val_s,   y_val_s,
                      X_test_s,  y_test_s)))