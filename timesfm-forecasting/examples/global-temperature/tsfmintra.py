import warnings
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


#global config
checkpoint_path = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LENGTH = 64 #User 64 points of history to predict the next point
HORIZON =1 #Predict the next point 1 day ahead
TRAIN_SIZE = 0.8 #Use 80% of the data for training and 20% for testing
VAL_SIZE = 0.1 #Use 10% of the data for validation
TEST_SIZE = 0.1 #Use 10% of the data for testing
TOTAL_SIZE=TRAIN_SIZE + VAL_SIZE + TEST_SIZE
FINETUNE_EPOCHS = 50
FINETUNE_LR = 1e-4
FINETUNE_BATCH=8
PATIENCE=15
OUTPUT_PATH="./timesfm_finetuned_model.pth"
DATE_COL="date"
HOUR_COL="hour"
DEBIT_COL="dr_amount"
Credit_COL="cr_amount"

FILE_PATH = r"C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting\examples\global-temperature\tsfm.xlsx"
SKIP_FINETUNE=False
FORCE_CPU=False
FREQ_CODE="0"


#Data helper 
def load_data(file_path: str | None) -> pd.DataFrame:
    if file_path:
        df = pd.read_excel(file_path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df[HOUR_COL] = df[DATE_COL].astype(int)
        df[DEBIT_COL] = pd.to_numeric(df[DEBIT_COL], errors='coerce').fillna(0)
        df[DEBIT_COL] =np.abs(df[DEBIT_COL])
        df[Credit_COL] = pd.to_numeric(df[Credit_COL], errors='coerce').fillna(0)
        #sort for determinism
        df = df.sort_values([DATE_COL, HOUR_COL]).reset_index(drop=True)
        return df
    else:
        raise ValueError("File path is required to load data.")
    
    def build_hour_series(df: pd.DataFrame,value_col:str,hour:int) -> pd.Series:
        '''extract a univariate series for a *single hour* across dates.Dataset
        Index:value_date(sorted)
        Values:df[value_col] for the given hour'''
        d=df.loc[df[HOUR_COL]==hour,[DATE_COL,value_col]].copy()
        d=d.sort_values(DATE_COL)
        s=pd.Series(d[value_col].to_numpy(),index=pd.to_datetime(d[DATE_COL])).to_numpy()
        s=s.astype(np.float32)
        #if there are duplicate dates for an hour , aggregrate sums to be safe
        if pd.Index(s.Index).has_duplicated().any():
            s=s.groupby(s.index).sum().astype(np.float32)
        return s
    
    def load_leon_data(loc):
        filename ='idl_model_kpi_{}.xlsx'.format(loc)
        kpi_file=fio.get_http_file_stream('lon','IDL/KPIs',filename)
        df=pd.read_excel(kpi_file,sheet_name='aggregated 1 min data',index_col=0)
        #aggregate to hourly data
        df=df.reset_index()
        df['ts']=pd.to_datetime(df['Business_Date'].astype(str)+' '+df['Hour'].astype(str)+" " + df["bucket"].astype(str))
        df=df.set_index('ts').sort_index()
        hourly=(df.resample("1H").agg({"credit_amount":"sum","debit_amount":"sum","credit_count":"sum","debit_count":"sum"}))
        #add more features
        df_feat=hourly.rename(columns={"credit_amount":"cr_amount","debit_amount":"dr_amount"})
        df_feat["transaction_count"]=(df_feat["credit_count"]+df_feat["debit_count"]).fillna(0)
        df_feat=df_feat.reset_index().rename(columns={"ts":"Datetime"})
        df_feat["date"]=df_feat["Datetime"].dt.date
        df_feat["hour"]=df_feat["Datetime"].dt.hour
        df_feat["day_of_week"]=df_feat["Datetime"].dt.dayofweek
        df_feat["day_name"]=df_feat["Datetime"].dt.day_name()
        return df_feat
    
    #2.Metrics
    def compute_metrics(actuals:np.ndarray, predictions:np.ndarray) -> dict:
        mae=mean_absolute_error(actuals, predictions)
        rmse=np.sqrt(mean_squared_error(actuals, predictions))
        mape=np.mean(np.abs((actuals - predictions) / actuals+1e-9)) * 100
        return{"MAE":round(mae, 4), "RMSE":round(rmse, 4), "MAPE%":round(mape, 4)}

        


    #3.TimesFM MODLE LOADER

    def _timesfm(FORCE_CPU:bool=False):
    """lOAD Timesfm via local saved file install:pip install timecopilot-timefm torch"""
    torch .set_float32_matmul_precision('high')

    #4.Rolling 1 day ahaed inference helper
    @torch.no_grad()
    def forecast(model,past_values,horizon,freq_code=0,device='"cpu"):'
        'x=torch
        

