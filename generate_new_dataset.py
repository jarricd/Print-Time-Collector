import random

from sklearn.mixture import GaussianMixture
import sqlalchemy
import datetime
from model.tables import DataEntry, Metadata, TrainingResults
import numpy
import pickle
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
import scipy.stats

if __name__ == "__main__":
    base_dset_conn = sqlalchemy.create_engine(f"sqlite:///3dprinting.db")
    target_db = sqlalchemy.create_engine(f"sqlite:///datasets/3dprinting_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.db", echo=True)
    base_x = []
    base_y = []

    with base_dset_conn.connect() as conn:
        print_time_entries = conn.execute(sqlalchemy.text("select distinct(print_time), object_size, object_name from data_entry;"))
    
    DataEntry.metadata.create_all(target_db)
    TrainingResults.metadata.create_all(target_db)

    for row in print_time_entries:
       listified_row = list(row)
       base_x.append([listified_row[1] / 1024])
       base_y.append(listified_row[0] / 3600)
    
    base_dset = numpy.column_stack((base_x, base_y))
    gmm = GaussianMixture(n_components=1)
    gmm.fit(base_dset)

    # save model
    with open(target_path := f"results/gmm_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pkl", "wb") as f:
        pickle.dump(gmm, f, protocol=pickle.HIGHEST_PROTOCOL)

    # todo add noise, so the samples spread better without a clear cutoff
    n_samples_per_batch = 5
    target_high_x_samples = 150
    new_samples = []
    while target_high_x_samples >= 0:
        generated_data = gmm.sample(n_samples_per_batch)
        target_mask = (generated_data[0][:, 0] > 5000) & (generated_data[0][:, 0] < 17500)
        high_file_size = generated_data[0][target_mask]
        generated_samples_count = len(high_file_size)
        if generated_samples_count > 0:
            target_high_x_samples -= generated_samples_count
            new_samples.append(numpy.round(high_file_size, 2))
    
    target_ultrahigh_x_samples = 150
    new_very_high_samples = []
    while target_ultrahigh_x_samples >= 0:
        generated_data = gmm.sample(n_samples_per_batch)
        target_mask = generated_data[0][:, 0] > 17500 
        high_file_size = generated_data[0][target_mask]
        generated_samples_count = len(high_file_size)
        if generated_samples_count > 0:
            target_ultrahigh_x_samples -= generated_samples_count
            new_very_high_samples.append(numpy.round(high_file_size, 2))

    new_samples = numpy.concatenate(new_samples, axis=0)
    new_very_high_samples = numpy.concatenate(new_very_high_samples, axis=0)
    full_dset = numpy.vstack((base_dset, new_samples, new_very_high_samples))
    augmented_vals = augment(full_dset)
    pass

    print("Saving dataset figure...")
    plt.figure(figsize=(10, 5))
    plt.scatter(base_dset[:, 0], base_dset[:, 1], label="Old dset")
    plt.scatter(new_samples[:, 0], new_samples[:, 1], label="Augmented dset - low_samples")
    plt.scatter(new_very_high_samples[:, 0], new_very_high_samples[:, 1], label="Augmented dset - high samples")
    plt.legend()
    plt.title("Dataset with generated data")
    plt.suptitle("Filesize vs print time")
    plt.xlabel("Object size [kB]")
    plt.ylabel("Print time [h]")
    plt.savefig(f"datasets/new_dset_visualisation_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.png")
    plt.show()

    print("Calculation coefficients...")
    pearson_coeff, pearson_pval = scipy.stats.pearsonr(full_dset[:, 0], full_dset[:, 1])
    print(f"Pearson coeff: {pearson_coeff}, pval: {pearson_pval}")
    spearman_coeff, spearman_pval = scipy.stats.spearmanr(full_dset[:, 0], full_dset[:, 1])
    print(f"Spearman coeff: {spearman_coeff}, pval: {spearman_pval}")

    with Session(target_db) as session:
        for x_sample, y_sample in full_dset:
            new_db_entry = DataEntry()
            new_db_entry.object_size = x_sample.item() 
            new_db_entry.print_time = y_sample.item() * 3600
            session.add(new_db_entry)

        dataset_metadata = Metadata()
        dataset_metadata.pearson_coefficient = pearson_coeff
        dataset_metadata.spearman_coefficient = spearman_coeff
        dataset_metadata.pearson_p_value = pearson_pval
        dataset_metadata.spearman_p_value = spearman_pval
        session.add(dataset_metadata)

        session.commit()