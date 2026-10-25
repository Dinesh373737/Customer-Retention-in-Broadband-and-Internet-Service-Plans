import type {Analysis,Job} from '../types';
export function recoverableJob(analysis?:Analysis):string{return analysis&&['queued','running','failed'].includes(analysis.status)?analysis.job_id??'':''}
export function jobBelongs(job:Job|undefined,analysisId:string,datasetId:string):boolean{return !!job&&job.analysis_id===analysisId&&(!job.dataset_id||job.dataset_id===datasetId)}
