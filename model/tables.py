from sqlalchemy import Float, String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass

class DataEntry(Base):
    __tablename__ = "data_entry"

    id: Mapped[int] = mapped_column(primary_key=True)
    print_time: Mapped[float] = mapped_column(Float(), default=0)
    estimated_time: Mapped[float] = mapped_column(Float(), default=0)
    print_time_left: Mapped[float] = mapped_column(Float(), default=0) #aka our GT
    print_progress: Mapped[float] = mapped_column(Float(), default=0)
    object_name: Mapped[String] = mapped_column(String(), default="")
    object_size: Mapped[int] = mapped_column(Integer(), default=0)


class Results(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(primary_key=True)
    predicted_print_time: Mapped[float] = mapped_column(Float())
    print_time_left: Mapped[float] = mapped_column(Float())
    model_type: Mapped[String] = mapped_column(String())
    model_path: Mapped[String] = mapped_column(String())

class TrainingResults(Base):
    __tablename__ = "training_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[String] = mapped_column(String(), unique=True)
    r2_score: Mapped[Float] = mapped_column(Float())
    avg_mse_error: Mapped[Float] = mapped_column(Float())
    avg_mae_error: Mapped[Float] = mapped_column(Float())


class Metadata(Base):
    __tablename__ = "metadata"
    id: Mapped[int] = mapped_column(primary_key=True)
    pearson_coefficient: Mapped[float] = mapped_column(Float())
    spearman_coefficient: Mapped[float] = mapped_column(Float())
    pearson_p_value: Mapped[float] = mapped_column(Float())
    spearman_p_value: Mapped[float] = mapped_column(Float())
