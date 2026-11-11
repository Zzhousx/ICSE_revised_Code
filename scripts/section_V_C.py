from pathlib import Path
import csv, io, sys, zipfile
import h5py
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def rows_for(dataset):
    with zipfile.ZipFile(ROOT/'data'/f'Dataset_{dataset}_raw.zip') as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith('-attributes.csv')):
            yield from csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig')))

def main():
    output = ROOT/'outputs'/'Section_V_C_quality.csv'
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['Dataset','ROC_AUC','Spearman_minus_TTC','Spearman_minus_DTO','N_TTC','N_DTO'])
        for dataset in (1,2,3):
            path=ROOT/'outputs'/'rankings'/f'Dataset_{dataset}_VoD-Sampler_seed00.mat'
            q=np.asarray(loadmat(path)['q']).ravel()
            with h5py.File(ROOT/'data'/f'Dataset_{dataset}.mat') as mat:
                y=mat['globalLabels'][:].ravel().astype(int)
            rows=list(rows_for(dataset))
            if not (len(q)==len(y)==len(rows)):raise ValueError('Feature rows and attribute rows differ.')
            vals=[]; counts=[]
            for column in ('Attribute[TTC]','Attribute[DTO]'):
                numeric=[]
                for row in rows:
                    try:numeric.append(float(row[column]))
                    except (KeyError,ValueError,TypeError):numeric.append(float('nan'))
                numeric=np.asarray(numeric);mask=np.isfinite(numeric)&np.isfinite(q)
                vals.append(spearmanr(q[mask],-numeric[mask]).statistic);counts.append(int(mask.sum()))
            result=['Dataset_'+str(dataset),roc_auc_score(y,q),*vals,*counts];writer.writerow(result);print(result[0],result[1:])

if __name__=='__main__':main()
