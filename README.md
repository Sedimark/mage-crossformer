# mage-crossformer
This project contains the wrap scrip for mage integration

## Key Features
- setup_fit
- inference

## Getting Started

> setup_fit function sample
```python
model, data, trainer = setup_fit(cfg, df)

"""
    Fit the model with the given configuration and data.

    Args:
        cfg (dict): Configuration dictionary.
        df (pd.DataFrame): DataFrame containing the data.
        callbacks (list, optional): List of callbacks to use during training. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the fitted model and the training history.
    """
```

> inference function sample
```python
predictions = inference(model, data):
    """
    Perform inference using the trained model on the provided data.

    Args:
        model (CrossFormer): The trained CrossFormer model.
        df (pd.DataFrame): DataFrame containing the data for inference.

    Returns:
        pd.DataFrame: DataFrame containing the predictions.
    """
```

## Additional Information
The wrap script will be cloned into the utils directory of the mage project. You can import and use these two functions in your pipelins. However, you need to start the fit or inference manually coperating the MLFLOW settings in pipelines.  


## Update

New python script "manipulation.py" is added to handle data manipulation tasks inside the mage blocks.

1) Class BaseManipulation
It provides the abstract methods for different data manipulation algorithm classes. Three common methods are included: train, inference, and prune. Each method needs to imlement its own logic.

2) Function initialize_manipulation
This function initializes the manipulation class based on the type name and configuration provided. It returns an instance of the corresponding manipulation class.

3) Example of a manipulation class implementation
MageCrossFormerManipulation is an example of a manipulation class that inherits from BaseManipulation. It implements the train, inference, and prune methods with specific logic for the CrossFormer model.

4) Example usage
```python
from mage_crossformer.manipulation import initialize_manipulation, MageCrossFormerManipulation

manipulation = initialize_manipulation("MageCrossFormerManipulation")

manipulation.train(cfg, df)
manipulation.inference(model, df)
```