import {useState} from 'react';
import {useWorkspace} from '../App';
import {api,errorMessage} from '../services/api';
import type {Dataset,Quickstart} from '../types';

export const ACCEPTED=/\.(csv|xlsx)$/i;
export const MAX_BYTES=25*1024*1024;

/** Upload (or sample) → automatic schema → analysis (+ training when labelled), with no form filling. */
export function useQuickstart(){
 const w=useWorkspace();const[busy,setBusy]=useState<''|'upload'|'configure'>('');const[error,setError]=useState('');const[result,setResult]=useState<Quickstart|null>(null);
 const finish=async(dataset:Dataset)=>{setBusy('configure');const started=await api.quickstart(dataset.id);await w.refresh();w.selectDataset(dataset.id);w.selectAnalysis(started.analysis_id);if(started.job_id)w.setJobId(started.job_id);setResult(started);w.toast(started.job_id?'Dataset configured automatically. Models are training now.':'Dataset configured automatically. No churn column was found, so it opened in exploratory mode.');w.go('analytics');return started};
 const run=async(source:()=>Promise<Dataset>)=>{setError('');setBusy('upload');try{return await finish(await source())}catch(e){setError(errorMessage(e))}finally{setBusy('')}};
 const upload=(file?:File,sheet?:string)=>{if(!file)return;if(!ACCEPTED.test(file.name)){setError('Use a CSV file or an Excel .xlsx workbook.');return}if(file.size>MAX_BYTES){setError('The upload limit is 25 MB. Please choose a smaller file.');return}return run(()=>api.upload(file,sheet||undefined))};
 const sample=()=>run(api.sample);
 return {upload,sample,busy,error,setError,result};
}
