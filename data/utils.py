import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import seaborn
import datetime 
from sklearn.model_selection import train_test_split
import sqlalchemy
from sqlalchemy import text

def get_session(db_path):
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    session = sessionmaker(bind=engine)
    return session()


def training_plot(context, n_neigh):
    # plot residual plot
    fig = plt.figure()
    fig.suptitle("Predictions and GT vs file size")
    ax = fig.add_subplot(111)
    ax.set_xlabel("File size [kB]")
    ax.set_ylabel("Predicted time [min]")
    ax.scatter(context["X_val"], context["Y_val"], label="GT")
    ax.scatter(context["X_val"], context[f"pred_{n_neigh}"], label="Predictions")
    sorted_x = sorted(context["X_val"])
    sorted_y = sorted(context[f"pred_{n_neigh}"])
    ax.plot(sorted_x, sorted_y, label="Prediction line")
    plt.legend(loc="upper left")
    curr_date_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    plt.savefig(f"training_plot_{curr_date_time}_{n_neigh}.png")
    plt.show()
    # mlflow.log_artifact(f"training_plot_{curr_date_time}_{n_neigh}.png")


def avg_metric_vs_n_neigh(item_type, context, neighs):
    fig = plt.figure()
    fig.suptitle(f"{item_type} vs n_neigh")
    ax = fig.add_subplot(111)
    ax.set_xlabel("n_neighbours")
    ax.set_ylabel(f"{item_type}")
    mse_values_list = []
    for n_neighbours in neighs:
        mse_values_list.append(context[f"{item_type.lower()}_avg_{n_neighbours}"])

    ax.scatter(neighs, mse_values_list)
    curr_date_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    plt.savefig(f"{item_type}_vs_n_neighs_plot_{curr_date_time}.png")
    plt.show()
    # mlflow.log_artifact(f"{item_type}_vs_n_neighs_plot_{curr_date_time}.png")

def mae_vs_file_size(context):
    if "file_size" not in context or "mae" not in context:
        raise ValueError("Context must contain 'file_size' and 'mae' keys")

        # Create a Seaborn scatter plot for MAE vs file size
    plt.figure(figsize=(10, 6))
    seaborn.scatterplot(x=context["file_size"], y=context["mae"])
    
    # Add titles and labels
    plt.title("MAE vs File Size")
    plt.xlabel("File Size [kB]")
    plt.ylabel("Mean Absolute Error (MAE)")

    # Save the plot to a file and display it
    plt.savefig("mae_vs_file_size.png")
    plt.show()

def get_dataset(db_path, split_size):
    db_conn = sqlalchemy.create_engine(f"sqlite:///{db_path}")

    with db_conn.connect() as conn:
        print_time_entries = conn.execute(text("select distinct(print_time), object_size, object_name from data_entry;"))

    context = {
        "mse": [],
        "mae": [],
        "X_val": [],
        "Y_val": [],
        "X": [],
        "Y": [],
        "models": [],
        "X_train": [],
        "Y_train": [],
        "scores": [],
        "pred": []
    }

    for row in print_time_entries:
        listified_row = list(row)
        context["X"].append([listified_row[1]])
        context["Y"].append(listified_row[0] / 60)


    X_train, X_val, Y_train, Y_val = train_test_split(context["X"], context["Y"], test_size=split_size)

    context["X_train"] = list(X_train)
    context["X_val"] = list(X_val)
    context["Y_train"] = list(Y_train)
    context["Y_val"] = list(Y_val)

    return context

def calculate_metrics(context, x_val, y_val, pred):
    mse_prediction_error = numpy.round(mean_squared_error(numpy.array([y_val]), pred), 3)
    mae_prediction_error = numpy.round(mean_absolute_error(numpy.array([y_val]), pred), 3)
    print(
        f"Input val: {x_val} kB, Target prediction: {y_val} min, Prediction: {pred} min,"
        f" MSE: {mse_prediction_error}, MAE: {mae_prediction_error}")
    context[f"mse"].append(mse_prediction_error)
    context[f"mae"].append(mae_prediction_error)
    context[f"pred"].append(pred.item())