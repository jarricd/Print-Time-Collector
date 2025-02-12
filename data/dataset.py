from sqlalchemy import distinct

import data.utils
import torch.utils.data
import sqlalchemy

from model.tables import DataEntry


class PrintTimeDataset(torch.utils.data.Dataset):
    def __init__(self, db_path):
        self.db_session = data.utils.get_session(db_path)

    def __len__(self):
        distinct_entries_stmt = sqlalchemy.select(distinct(DataEntry.print_time), DataEntry.object_size, DataEntry.object_name)
        print_time_entries = self.db_session.execute(distinct_entries_stmt).all()
        return len(print_time_entries)


    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    dset = PrintTimeDataset('../3dprinting.db')
    print(len(dset))