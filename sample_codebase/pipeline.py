from model import load_data, preprocess, train_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split

def run_pipeline(filepath):
    print("Loading data")
    df = load_data(filepath)
    
    print("Preprocessing...")
    X, y = preprocess(df)
    
    print("Training model")
    model, X_test, y_test = train_model(X, y)
    
    print("Evaluating...")
    rmse, acc = evaluate_model(model, X_test, y_test)
    
    print("Saving model")
    save_model(model)
    
    return model, rmse, acc

def debug_pipeline(filepath):
    try:
        run_pipeline(filepath)
    except KeyError as e:
        print(f"KeyError: Column {e} not found. Check your column names.")
    except ValueError as e:
        print(f"ValueError: {e}. Check data types and missing values.")
    except FileNotFoundError:
        print("File not found. Check your filepath.")

if __name__ == "__main__":
    run_pipeline("data.csv")