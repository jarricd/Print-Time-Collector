import seaborn
import sqlalchemy
import click
from sqlalchemy import create_engine, distinct
from sqlalchemy.orm import sessionmaker

from model.tables import DataEntry
seaborn.set_theme(style="white")

@click.group()
def cli():
    pass


@click.command()
@click.argument("db_path")
def plot_object_size_vs_print_time(db_path):
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    session = sessionmaker(bind=engine)()
    select_stmt_data_object_size = sqlalchemy.select(distinct(DataEntry.object_name), DataEntry.object_size, DataEntry.print_time)
    print_time_entries = session.execute(select_stmt_data_object_size).all()
    dataset_dict = {
        "object_size [kB]": [],
        "print_time [h]": []
    }
    for row in print_time_entries:
        listifed_row = list(row)
        dataset_dict["object_size [kB]"].append(listifed_row[1] / 1024)
        dataset_dict["print_time [h]"].append(listifed_row[2] / 3600)

    plot = seaborn.relplot(x="object_size [kB]", y='print_time [h]', data=dataset_dict, sizes=(50, 200), height=8)
    plot.figure.savefig("visualisation.png")
    


@cli.command()
@click.argument("db_path")
def train_knr_model(db_path):
    mlflow.set_tracking_uri(uri="http://192.168.0.67:8085")
    mlflow.set_experiment("KNR_Print_Time_Prediction")
    session = get_session(db_path)
    n_neighbors = 2
    kn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    select_stmt_data_object_size = sqlalchemy.select(distinct(DataEntry.object_name), DataEntry.object_size,
                                                     DataEntry.print_time)
    print_time_entries = session.execute(select_stmt_data_object_size).all()
    context = {
        "mse": [],
        "mae": [],
        "pred": [],
        "X_val": [],
        "Y_val": [],
        "X": [],
        "Y": [],
        "model": None,
        "X_train": [],
        "Y_train": [],
        "scores": []
    }

    for row in print_time_entries:
        listified_row = list(row)
        context["X"].append([listified_row[1] / 1024])
        context["Y"].append(listified_row[2] / 60)

    X_train, X_val, Y_train, Y_val = train_test_split(context["X"], context["Y"], test_size=0.25)
    context["X_train"] = list(X_train)
    context["X_val"] = list(X_val)
    context["Y_train"] = list(Y_train)
    context["Y_val"] = list(Y_val)
    kn_regressor.fit(numpy.array(X_train), numpy.array(Y_train))
    r2_score = kn_regressor.score(X_val, Y_val)
    click.echo(f"Regression score for validation set: {r2_score}")
    with mlflow.start_run():
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("R_2 score", r2_score)
        step = 0
        for x_val_datapoint, y_val_datapoint in zip(X_val, Y_val):
            predicted_value = numpy.round(kn_regressor.predict(numpy.array([x_val_datapoint])), 3)
            mse_prediction_error = numpy.round(mean_squared_error(numpy.array([y_val_datapoint]), predicted_value), 3)
            mae_prediction_error = numpy.round(mean_absolute_error(numpy.array([y_val_datapoint]), predicted_value), 3)
            click.echo(f"Input val: {x_val_datapoint} kB, Target prediction: {y_val_datapoint} min, Prediction: {predicted_value} min,"
                       f" MSE: {mse_prediction_error}, MAE: {mae_prediction_error}")
            context["mse"].append(mse_prediction_error)
            context["mae"].append(mae_prediction_error)
            context["pred"].append(predicted_value.item())
            mlflow.log_metric("mse", mse_prediction_error, step)
            mlflow.log_metric("mae", mae_prediction_error, step)
            step += 1

        # we dont want lists
        context["X_val"] = [item[0] for item in context["X_val"]]
        training_plot(context)
        context["model"] = kn_regressor
        with open(target_path := f"results/{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pkl", "wb") as f:
            pickle.dump(context, f, protocol=pickle.HIGHEST_PROTOCOL)

        mlflow.log_artifact(target_path)
        mlflow.log_artifact("training_plot.png")


if __name__ == "__main__":
    cli()