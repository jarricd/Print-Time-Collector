def train_xgboost_classifier(db_path):
    db_conn = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    TrainingResults.metadata.create_all(db_conn)
    with db_conn.connect() as conn:
        print_time_entries = conn.execute(
            text("select distinct(print_time), object_size, object_name from data_entry;"))

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
    n_estimators = [2]
    max_depth =

if __name__ == "__main__":
    pass