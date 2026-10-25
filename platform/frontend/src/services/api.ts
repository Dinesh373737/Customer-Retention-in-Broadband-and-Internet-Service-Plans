import type { Analysis, Configuration, Dataset, Job, Row } from '../types';
export const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1';
export class ApiError extends Error {status:number; constructor(message:string,status:number){super(message);this.name='ApiError';this.status=status}}
export function errorMessage(value:unknown):string {if(value instanceof Error)return value.message;if(typeof value==='string')return value;return 'An unexpected error occurred.'}
export async function request<T=Row>(path:string,init:RequestInit={}):Promise<T>{
  const headers=new Headers(init.headers);if(init.body&&!(init.body instanceof FormData))headers.set('Content-Type','application/json');
  const response=await fetch(`${API_BASE}${path}`,{...init,headers});
  if(!response.ok){let message=`Request failed (${response.status})`;try{const body=await response.json();const d=body.detail??body.error??body.message;message=typeof d==='string'?d:Array.isArray(d)?d.map((x:Row)=>`${x.loc?.slice(1).join('.')??''}: ${x.msg??JSON.stringify(x)}`).join('; '):JSON.stringify(d)}catch{/* Preserve useful HTTP status when body is not JSON. */}throw new ApiError(message,response.status)}
  if(response.status===204)return undefined as T;return response.json() as Promise<T>;
}
export function params(values:Record<string,unknown>){const p=new URLSearchParams();for(const[k,v]of Object.entries(values))if(v!==undefined&&v!==null&&v!=='')p.set(k,typeof v==='object'?JSON.stringify(v):String(v));return p.toString()?`?${p}`:''}
export const api={
  datasets:()=>request<{datasets:Dataset[]}>('/datasets'),analyses:()=>request<{analyses:Analysis[]}>('/analyses'),
  dataset:(id:string)=>request<Dataset>(`/datasets/${id}/profile`),
  upload:(file:File,sheet?:string)=>{const data=new FormData();data.set('file',file);if(sheet)data.set('sheet_name',sheet);return request<Dataset>('/datasets/upload',{method:'POST',body:data})},
  configure:(id:string,configuration:Configuration)=>request<Row>(`/datasets/${id}/configuration`,{method:'POST',body:JSON.stringify(configuration)}),
  analyze:(dataset_id:string)=>request<Analysis>('/analyses',{method:'POST',body:JSON.stringify({dataset_id})}),
  train:(dataset_id:string,fast:boolean,selection_metric:string,random_state:number)=>request<{analysis_id:string;job_id:string;status:string}>('/analyses/train',{method:'POST',body:JSON.stringify({dataset_id,fast,selection_metric,random_state})}),
  job:(id:string)=>request<Job>(`/jobs/${id}`),
  analysis:(id:string,endpoint:string,query:Record<string,unknown>={})=>request<Row>(`/analyses/${id}/${endpoint}${params(query)}`),
  post:(id:string,endpoint:string,body:unknown)=>request<Row>(`/analyses/${id}/${endpoint}`,{method:'POST',body:JSON.stringify(body)}),
  batch:(id:string,file:File,model?:string,sheet?:string)=>{const data=new FormData();data.set('file',file);if(model)data.set('model',model);if(sheet)data.set('sheet_name',sheet);return request<Row>(`/analyses/${id}/batch-score`,{method:'POST',body:data})},
  exportUrl:(id:string,kind:string,query:Record<string,unknown>={})=>`${API_BASE}/analyses/${id}/export${params({kind,...query})}`,
};
export function downloadRows(rows:Row[],filename:string){if(!rows.length)return;const columns=Array.from(new Set(rows.flatMap(r=>Object.keys(r))));const cell=(value:unknown)=>{let text=value==null?'':typeof value==='object'?JSON.stringify(value):String(value);if(/^[=+@-]/.test(text))text=`'${text}`;return `"${text.replace(/"/g,'""')}"`};const csv=[columns.map(cell).join(','),...rows.map(row=>columns.map(c=>cell(row[c])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob(['\ufeff',csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url)}
