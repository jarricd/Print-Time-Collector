import sqlalchemy
import requests
import logging
import sys

from model.config import DATABASE_FILE
from model.tables import DataEntry

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logging.getLogger().addHandler(fh:=logging.FileHandler("collector.log"))
logging.getLogger().addHandler(sh:=logging.StreamHandler())

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


if __name__ == "__main__":
    engine = create_engine(f"sqlite:///{DATABASE_FILE}", echo=True)
    DataEntry.metadata.create_all(engine)
    try:
        job_req = requests.get("http://192.168.0.67:5000/api/job", headers={"X-Api-Key": "75EB947ECFF141D3A85B91F8ACC8B2D0"})
        job_req.raise_for_status()
    except Exception as e:
        logging.error(f"Exception occured to GET a job status failed. Exc msg {e}")
        sys.exit(1)

    job_req_response = job_req.json()
    
    if "error" not in job_req_response:
        if job_req_response.get("progress").get("completion") == 100:
            new_entry = DataEntry(print_time=job_req_response["progress"]["printTime"],
                                estimated_time=job_req_response["job"]["estimatedPrintTime"],
                                print_time_left=job_req_response["progress"]["printTimeLeft"],
                                print_progress=job_req_response["progress"]["completion"],
                                object_name=job_req_response["job"]["file"]["path"],
                                object_size=job_req_response["job"]["file"]["size"]
                                )
            
            with Session(engine) as session:
                session.add(new_entry)
                session.commit()
            logging.info("Data added.")
        else:
            logging.info("Job is not completed, will retry later.")
    else:
        logging.warning("Job not ready, check printer. Exiting.")
        sys.exit(2)

    sys.exit(0)
