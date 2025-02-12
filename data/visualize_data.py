import seaborn
import sqlalchemy
import click
from sqlalchemy import create_engine, distinct
from sqlalchemy.orm import sessionmaker

from model.tables import DataEntry
seaborn.set_theme(style="white")


@click.command()
@click.argument("db_path")
def plot_object_size_vs_print_time(db_path):
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    session = sessionmaker(bind=engine)()
    pass
    # select_stmt_data_object_size = sqlalchemy.select(distinct(DataEntry.object_name), DataEntry.object_size, DataEntry.print_time)
    # print_time_entries = session.execute(select_stmt_data_object_size).all()
    # dataset_dict = {
    #     "object_size [kB]": [],
    #     "print_time [h]": []
    # }
    # for row in print_time_entries:
    #     listifed_row = list(row)
    #     dataset_dict["object_size [kB]"].append(listifed_row[1] / 1024)
    #     dataset_dict["print_time [h]"].append(listifed_row[2] / 3600)
    #
    # plot = seaborn.relplot(x="object_size [kB]", y='print_time [h]', data=dataset_dict, sizes=(50, 200), height=8)
    # plot.figure.savefig("visualisation.png")

if __name__ == "__main__":
    plot_object_size_vs_print_time()