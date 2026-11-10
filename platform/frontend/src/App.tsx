import {createContext,useContext,useEffect,useState,type ReactNode} from 'react';
import {useQuery,useQueryClient} from '@tanstack/react-query';
import {ArrowUpRight,BarChart3,Database,FileText,FlaskConical,Home,Layers3,Lightbulb,LoaderCircle,Menu,Target,Users,X} from 'lucide-react';
import {api,ApiError,errorMessage} from './services/api';
import {jobBelongs} from './features/jobs';
import type {Analysis,Dataset,Page,Row} from './types';
import {pageNames} from './types';
import {recoverableJob} from './features/jobs';
import {Select} from './components/ui';
import {Workspace,Studio} from './pages/Studio';
import {Analytics,ModelLab,Explainability} from './pages/Analysis';
import {Customers} from './pages/Customers';
import {Strategy,Reports} from './pages/Strategy';

interface ContextValue {page:Page;go:(p:Page)=>void;datasetId:string;analysisId:string;dataset?:Dataset;analysis?:Analysis;datasets:Dataset[];analyses:Analysis[];selectDataset:(id:string)=>void;selectAnalysis:(id:string)=>void;model:string;setModel:(model:string)=>void;models?:Row;modelsLoading:boolean;modelsError:unknown;refresh:()=>Promise<void>;toast:(s:string,tone?:'success'|'error')=>void;jobId:string;setJobId:(id:string)=>void}
const notified=new Set<string>();
const WorkspaceContext=createContext<ContextValue|null>(null);
export function useWorkspace(){const value=useContext(WorkspaceContext);if(!value)throw new Error('Workspace context is missing');return value}
function saved(key:string){try{return localStorage.getItem(`retention2.${key}`)??''}catch{return ''}}
function persist(key:string,value:string){try{localStorage.setItem(`retention2.${key}`,value)}catch{/* Session remains functional without local storage. */}}
const navGroups:{label:string;items:[Page,typeof Layers3][]}[]=[
 {label:'Data',items:[['workspace',Home],['studio',Database]]},
 {label:'Analysis',items:[['analytics',BarChart3],['models',FlaskConical],['explain',Lightbulb]]},
 {label:'Action',items:[['customers',Users],['strategy',Target],['reports',FileText]]},
];
export default function App(){
 const qc=useQueryClient();const[page,setPage]=useState<Page>((Object.keys(pageNames).includes(location.hash.slice(1))?location.hash.slice(1):'workspace') as Page);const[datasetId,setDatasetId]=useState(saved('dataset'));const[analysisId,setAnalysisId]=useState(saved('analysis'));const[model,setModel]=useState('');const[jobId,setJobIdState]=useState(saved('job'));const[mobile,setMobile]=useState(false);const[message,setMessage]=useState<{text:string;tone:string}|null>(null);
 const datasetsQ=useQuery({queryKey:['datasets'],queryFn:api.datasets});const analysesQ=useQuery({queryKey:['analyses'],queryFn:api.analyses});const datasets=datasetsQ.data?.datasets??[];const analyses=analysesQ.data?.analyses??[];
 const datasetQ=useQuery({queryKey:['dataset',datasetId],queryFn:()=>api.dataset(datasetId),enabled:!!datasetId});const dataset=datasetQ.data;const analysis=analyses.find(a=>(a.analysis_id??a.id)===analysisId);
 const modelsQ=useQuery({queryKey:['models',analysisId],queryFn:()=>api.analysis(analysisId,'models'),enabled:!!analysisId,retry:(count,error)=>!(error instanceof ApiError&&[404,409].includes(error.status))&&count<1});
 useEffect(()=>{persist('dataset',datasetId)},[datasetId]);useEffect(()=>{persist('analysis',analysisId);setModel('')},[analysisId]);useEffect(()=>{persist('job',jobId)},[jobId]);
 useEffect(()=>{if(modelsQ.data&&!model){setModel(modelsQ.data.preferred_model??modelsQ.data.selected_model??'')}},[modelsQ.data,model]);
 useEffect(()=>{if(!message)return;const timer=setTimeout(()=>setMessage(null),5500);return()=>clearTimeout(timer)},[message]);
 useEffect(()=>{if(datasetQ.error instanceof ApiError&&datasetQ.error.status===404){setDatasetId('');setAnalysisId('')}},[datasetQ.error]);
 useEffect(()=>{if(!analysisId){setJobIdState('');return}if(!analysis)return;const recovered=recoverableJob(analysis);if(recovered&&recovered!==jobId)setJobIdState(recovered);else if(jobId&&analysis.job_id!==jobId)setJobIdState('')},[analysisId,analysis?.job_id,analysis?.status]);
 // With nothing saved in this browser, open the most recent analysis instead of an empty workspace.
 useEffect(()=>{if(datasetId||analysisId||!analyses.length)return;const latest=[...analyses].sort((a,b)=>String(b.created_at??'').localeCompare(String(a.created_at??'')))[0];if(latest){setAnalysisId(latest.analysis_id??latest.id??'');setDatasetId(latest.dataset_id);setJobIdState(recoverableJob(latest))}},[analyses.length]);
 const go=(next:Page)=>{setPage(next);location.hash=next;setMobile(false)};
 const selectDataset=(id:string)=>{setDatasetId(id);const latest=analyses.find(a=>a.dataset_id===id);setAnalysisId(latest?.analysis_id??latest?.id??'');setJobIdState(recoverableJob(latest))};
 const selectAnalysis=(id:string)=>{setAnalysisId(id);const a=analyses.find(a=>(a.analysis_id??a.id)===id);if(a)setDatasetId(a.dataset_id);setJobIdState(recoverableJob(a))};
 const refresh=async()=>{await Promise.all([qc.invalidateQueries({queryKey:['datasets']}),qc.invalidateQueries({queryKey:['dataset']}),qc.invalidateQueries({queryKey:['analyses']}),qc.invalidateQueries({queryKey:['models']}),qc.invalidateQueries({queryKey:['analysis']})])};
 const ctx:ContextValue={page,go,datasetId,analysisId,dataset,analysis,datasets,analyses,selectDataset,selectAnalysis,model,setModel,models:modelsQ.data,modelsLoading:modelsQ.isFetching,modelsError:modelsQ.error,refresh,toast:(text,tone='success')=>setMessage({text,tone}),jobId,setJobId:setJobIdState};
 const pages:Record<Page,ReactNode>={workspace:<Workspace/>,studio:<Studio/>,analytics:<Analytics/>,models:<ModelLab/>,explain:<Explainability/>,customers:<Customers/>,strategy:<Strategy/>,reports:<Reports/>};
 const job=useQuery({queryKey:['job',jobId],queryFn:()=>api.job(jobId),enabled:!!jobId,refetchInterval:q=>q.state.data&&['completed','failed'].includes(q.state.data.status)?false:1200});
 const training=jobBelongs(job.data,analysisId,datasetId)&&job.data&&['queued','running'].includes(job.data.status)?job.data:undefined;
 useEffect(()=>{const j=job.data;if(!j||!jobBelongs(j,analysisId,datasetId)||notified.has(j.id))return;if(j.status==='completed'){notified.add(j.id);void refresh();setMessage({text:'Training finished. Predictions, explanations and the retention plan are ready.',tone:'success'})}else if(j.status==='failed'){notified.add(j.id);void refresh();setMessage({text:'Training failed. Open Models for details.',tone:'error'})}},[job.data?.id,job.data?.status]);
 const versions=analyses.filter(a=>a.dataset_id===datasetId);
 return <WorkspaceContext.Provider value={ctx}><div className="app-shell"><a className="skip-link" href="#main-content">Skip to main content</a>{mobile&&<div className="sidebar-scrim" onClick={()=>setMobile(false)}/>}
 <aside className={`sidebar ${mobile?'sidebar-open':''}`}>
  <a className="brand" href="#workspace" onClick={()=>go('workspace')}><span className="brand-mark" aria-hidden="true"><Layers3 size={16}/></span><span className="brand-name">Retention Lab</span></a>
  <button className="mobile-close icon-button" aria-label="Close navigation" onClick={()=>setMobile(false)}><X size={18}/></button>
  <nav aria-label="Main navigation">{navGroups.map(group=><div className="nav-group" key={group.label}><div className="sidebar-label">{group.label}</div>{group.items.map(([key,Icon])=><button className={`nav-item ${page===key?'active':''}`} key={key} onClick={()=>go(key)} aria-current={page===key?'page':undefined}><Icon size={16}/><span>{pageNames[key]}</span></button>)}</div>)}</nav>
  <div className="sidebar-bottom"><a href="http://127.0.0.1:8501" target="_blank" rel="noreferrer" className="legacy-link">IBM Telco case study<ArrowUpRight size={14}/></a><span className="sidebar-footer">Runs locally · no customer outreach</span></div>
 </aside>
 <div className="main-shell"><header className="topbar"><div className="topbar-left"><button className="icon-button mobile-toggle" onClick={()=>setMobile(true)} aria-label="Open navigation"><Menu size={19}/></button><h2 className="topbar-title">{pageNames[page]}</h2></div>
  <div className="topbar-controls">
   <div className="topbar-field"><label htmlFor="workspace-dataset">Dataset</label><Select id="workspace-dataset" value={datasetId} onChange={e=>selectDataset(e.target.value)}><option value="">Select a dataset</option>{datasets.map(d=><option value={d.id} key={d.id}>{d.filename}</option>)}</Select></div>
   {!!analysisId&&versions.length>1&&<div className="topbar-field"><label htmlFor="analysis-version">Version</label><Select id="analysis-version" value={analysisId} onChange={e=>selectAnalysis(e.target.value)}>{versions.map(a=><option key={a.analysis_id??a.id} value={a.analysis_id??a.id}>{a.created_at?new Date(a.created_at).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):(a.analysis_id??a.id??'').slice(0,8)} · {a.mode==='historical'?(a.status==='completed'?'trained':a.status):'exploratory'}</option>)}</Select></div>}
   {model&&<div className="topbar-field"><label htmlFor="prediction-model">Model</label><Select id="prediction-model" value={model} onChange={e=>setModel(e.target.value)}>{(modelsQ.data?.models??[]).map((name:string)=><option key={name}>{name}</option>)}</Select></div>}
  </div></header>
 <main id="main-content" tabIndex={-1}>
 {training&&<div className="job-banner" role="status"><LoaderCircle size={15} className="animate-spin"/><span><strong>Training models</strong> · {training.message}</span><div className="job-banner-track"><div style={{width:`${Math.max(4,Math.min(100,training.progress))}%`}}/></div><span className="mono-num">{training.progress}%</span></div>}
 {(datasetsQ.isError||analysesQ.isError)&&<div className="connection-error" role="alert"><strong>Backend unavailable.</strong> Start the FastAPI service on port 8000. <button onClick={()=>{void datasetsQ.refetch();void analysesQ.refetch()}}>Retry connection</button><small>{errorMessage(datasetsQ.error??analysesQ.error)}</small></div>}
 <div className="page">{pages[page]}</div></main></div>
 {message&&<div className={`toast toast-${message.tone}`} role="status"><span>{message.text}</span><button aria-label="Dismiss notification" onClick={()=>setMessage(null)}><X size={15}/></button></div>}</div></WorkspaceContext.Provider>
}
