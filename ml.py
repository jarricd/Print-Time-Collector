import datetime

import pandas
import seaborn
import sqlalchemy
from sklearn.model_selection import train_test_split
from sqlalchemy import distinct, text
from sklearn.neighbors import KNeighborsRegressor
import numpy
from sqlalchemy.orm import Session

import data
from data.utils import avg_metric_vs_n_neigh, get_session, training_plot, calculate_metrics
from model.tables import DataEntry, TrainingResults
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from sklearn.neural_network import MLPRegressor

seaborn.set_theme(style="white")



def plot_object_size_vs_print_time(db_path):
    # session = get_session(db_path)
    db_conn = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    dataset_dict = {
        "object_size [kB]": [],
        "print_time [min]": []
    }
    with db_conn.connect() as conn:
        result = conn.execute(text("select distinct(print_time), object_size, object_name from data_entry;"))
        for row in result:
            dataset_dict["print_time [min]"].append(row[0] / 60)
            dataset_dict["object_size [kB]"].append(row[1])

    plot = seaborn.relplot(x="object_size [kB]", y='print_time [min]', data=dataset_dict, sizes=(50, 300), height=8)
    plot.figure.savefig("visualisation.png")

def train_mlp_regressor(db_path):
    hidden_layer_size = (10,10) 
    context = data.utils.get_dataset(db_path, 0.33)
    max_iter = 1000
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, max_iter=max_iter, random_state=9)
    model.fit(context["X_train"], context["Y_train"])
    for x_val, y_val in zip(context["X_val"], context["Y_val"]):
        pred = model.predict([x_val])
        calculate_metrics(context, x_val, y_val, pred)

    x_val_without_lists = [val[0] for val in context["X_val"]]
    # prepare pandas df
    output_context = {}
    output_context.update({"mse": context["mse"]})
    output_context.update({"mae": context["mae"]})
    output_context.update({"X_val": x_val_without_lists})
    output_context.update({"Y_val": context["Y_val"]})
    output_context.update({"pred": context["pred"]})

    pandas_df = pandas.DataFrame(output_context)
    pandas_df.to_excel("output.xlsx")




def train_knr_model(db_path):
    # mlflow.set_tracking_uri("http://192.168.0.67:8085")
    # mlflow.set_experiment("KNR_Print_Time_Prediction")
    n_neighbours_list = [2, 4, 5, 8, 10, 20]
   
    db_conn = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    TrainingResults.metadata.create_all(db_conn)
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
        "scores": []
    }

    for row in print_time_entries:
        listified_row = list(row)
        context["X"].append([listified_row[1]])
        context["Y"].append(listified_row[0] / 60)


    X_train, X_val, Y_train, Y_val = train_test_split(context["X"], context["Y"], test_size=0.33)

    context["X_train"] = list(X_train)
    context["X_val"] = list(X_val)
    context["Y_train"] = list(Y_train)
    context["Y_val"] = list(Y_val)

    # with mlflow.start_run():
    for n_neigbours in n_neighbours_list:
        print(f"Fitting for N neigh: {n_neigbours}")
        kn_regressor = KNeighborsRegressor(n_neighbors=n_neigbours, weights="distance", p=2)
        kn_regressor.fit(numpy.array(X_train), numpy.array(Y_train))
        context.update({f"pred_{n_neigbours}": []})
        context.update({f"mse_{n_neigbours}": []})
        context.update({f"mae_{n_neigbours}": []})
        context.update({f"r2_{n_neigbours}": -1})

        step = 0

        for x_val_datapoint, y_val_datapoint in zip(X_val, Y_val):
            predicted_value = numpy.round(kn_regressor.predict(numpy.array([x_val_datapoint])), 3)
            mse_prediction_error = numpy.round(mean_squared_error(numpy.array([y_val_datapoint]), predicted_value), 3)
            mae_prediction_error = numpy.round(mean_absolute_error(numpy.array([y_val_datapoint]), predicted_value), 3)
            print(f"Input val: {x_val_datapoint} kB, Target prediction: {y_val_datapoint} min, Prediction: {predicted_value} min,"
                    f" MSE: {mse_prediction_error}, MAE: {mae_prediction_error}")
            context[f"mse_{n_neigbours}"].append(mse_prediction_error)
            context[f"mae_{n_neigbours}"].append(mae_prediction_error)
            context[f"pred_{n_neigbours}"].append(predicted_value.item())
            # mlflow.log_metric(f"mse_{n_neigbours}", mse_prediction_error, step)
            # mlflow.log_metric(f"mae_{n_neigbours}", mae_prediction_error, step)
            
            step += 1

        r2_y_val = [[item] for item in Y_val]
        r2_pred_val = [[item] for item in context[f"pred_{n_neigbours}"]]
        r2_score_val = r2_score(numpy.array(r2_y_val), numpy.array(r2_pred_val))
        # mlflow.log_metric("R_2 score", r2_score, n_neigbours)
        print(f"R2 Score of regression is: {r2_score_val}")
        context[f"r2_{n_neigbours}"] = r2_score(numpy.array(r2_y_val), numpy.array(r2_pred_val))

        mse_mean = numpy.array(context[f"mse_{n_neigbours}"]).mean()
        context.update({f"mse_avg_{n_neigbours}": mse_mean})
        print(f"AVG mse for {n_neigbours}: {mse_mean}")
        
        mae_mean = numpy.array(context[f"mae_{n_neigbours}"]).mean()
        context.update({f"mae_avg_{n_neigbours}": mae_mean})
        print(f"AVG mae for {n_neigbours}: {mae_mean}")
        
        # we dont want lists
        context["X_val"] = [item for item in context["X_val"]]
        training_plot(context, n_neigbours)
        context["models"].append(kn_regressor)

        with open(target_path := f"results/{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_{n_neigbours}.pkl",
                  "wb") as f:
            pickle.dump(context, f, protocol=pickle.HIGHEST_PROTOCOL)


        with get_session(db_path) as session:
            # first get all previous results
            target_model_name = f"knr_{n_neigbours}"
            select_stmt = sqlalchemy.select(TrainingResults).where(TrainingResults.model_name==target_model_name)
            training_results_for_n_neighs = session.execute(select_stmt).first()
            if not training_results_for_n_neighs:
                # mlflow.log_artifact(target_path)
                training_results = TrainingResults()
                training_results.model_name = target_model_name
                training_results.avg_mae_error = context[f"mae_avg_{n_neigbours}"]
                training_results.avg_mse_error = context[f"mse_avg_{n_neigbours}"]
                training_results.r2_score = context[f"r2_{n_neigbours}"]
                session.add(training_results)
                session.commit()
            else:
                training_results_for_n_neighs[0].avg_mae_error = context[f"mae_avg_{n_neigbours}"]
                training_results_for_n_neighs[0].avg_mse_error = context[f"mse_avg_{n_neigbours}"]
                training_results_for_n_neighs[0].r2_score = context[f"r2_{n_neigbours}"]
                session.commit()


    avg_metric_vs_n_neigh("MSE", context, n_neighbours_list)
    avg_metric_vs_n_neigh("MAE", context, n_neighbours_list)





if __name__ == "__main__":
    target_db = "datasets/3dprinting_02_07_2025_14_22_40.db"
    plot_object_size_vs_print_time(target_db)
    train_knr_model(target_db)
    # train_mlp_regressor("datasets/3dprinting_02_05_2025_14_27_50.db")