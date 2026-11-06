import {useEffect,useRef,useState} from 'react';
import {useQuery} from '@tanstack/react-query';
import {ChevronDown,Search} from 'lucide-react';
import {useWorkspace} from '../App';
import {api} from '../services/api';
import type {Row} from '../types';
import {RiskBadge} from './ui';

/** Search customers by reference instead of typing a row index. Results are ordered by predicted risk. */
export function CustomerPicker({value,onChange,allowAll=false,label='Customer',record}:{value?:number;onChange:(row?:number,record?:Row)=>void;allowAll?:boolean;label?:string;record?:Row}){
 const w=useWorkspace();const[open,setOpen]=useState(false);const[search,setSearch]=useState('');const[query,setQuery]=useState('');const[picked,setPicked]=useState<Row|null>(null);const ref=useRef<HTMLDivElement>(null);
 useEffect(()=>{const t=setTimeout(()=>setQuery(search),250);return()=>clearTimeout(t)},[search]);
 useEffect(()=>{if(!open)return;const close=(e:MouseEvent)=>{if(ref.current&&!ref.current.contains(e.target as Node))setOpen(false)};document.addEventListener('mousedown',close);return()=>document.removeEventListener('mousedown',close)},[open]);
 useEffect(()=>{if(value===undefined)setPicked(null)},[value]);
 const results=useQuery({queryKey:['analysis',w.analysisId,'customer-picker',w.model,query],queryFn:()=>api.analysis(w.analysisId,'customers',{model:w.model,search:query,limit:8,sort:'desc'}),enabled:!!w.analysisId&&open});
 const choose=(record?:Row)=>{setPicked(record??null);onChange(record?.row_index,record);setOpen(false);setSearch('')};
 const shown=picked&&picked.row_index===value?picked:record&&record.row_index===value?record:null;
 const display=value===undefined?(allowAll?'All customers':'Choose a customer'):shown?shown.customer_reference:`Record ${value+1}`;
 return <div className="picker" ref={ref}><span className="picker-label">{label}</span><button type="button" className="picker-button" aria-haspopup="listbox" aria-expanded={open} aria-label={`${label}: ${display}`} onClick={()=>setOpen(!open)}><span>{display}</span>{typeof shown?.churn_probability==='number'&&<span className="picker-meta">{(shown.churn_probability*100).toFixed(0)}% risk</span>}<ChevronDown size={15}/></button>
 {open&&<div className="picker-panel"><div className="picker-search"><Search size={15}/><input autoFocus aria-label="Search customers" placeholder="Search by customer reference" value={search} onChange={e=>setSearch(e.target.value)}/></div><ul role="listbox">{allowAll&&<li><button type="button" role="option" aria-selected={value===undefined} onClick={()=>choose(undefined)}>All customers<small>Segment-level view</small></button></li>}
 {results.isFetching&&!results.data?<li className="picker-empty">Searching…</li>:(results.data?.rows??[]).map((r:Row)=><li key={r.row_index}><button type="button" role="option" aria-selected={r.row_index===value} onClick={()=>choose(r)}><span>{r.customer_reference}</span><span className="picker-row-meta">{typeof r.churn_probability==='number'&&<span className="mono-num">{(r.churn_probability*100).toFixed(1)}%</span>}<RiskBadge value={r.risk_category}/></span></button></li>)}
 {results.data&&!results.data.rows?.length&&<li className="picker-empty">No matching customers</li>}</ul><div className="picker-foot">Sorted by predicted churn, highest first</div></div>}</div>
}

/** Row index of the highest-risk customer, used as a sensible default selection. */
export function useTopCustomer(){
 const w=useWorkspace();
 const top=useQuery({queryKey:['analysis',w.analysisId,'top-customer',w.model],queryFn:()=>api.analysis(w.analysisId,'customers',{model:w.model,limit:1,sort:'desc'}),enabled:!!w.analysisId&&!!w.models});
 return top.data?.rows?.[0] as Row|undefined;
}
