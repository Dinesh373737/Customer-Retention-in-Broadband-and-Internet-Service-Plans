export type Row = Record<string, any>;
export type FeatureKind = 'numeric'|'categorical'|'ordinal'|'datetime';
export type Domain = 'telecommunications'|'banking'|'ecommerce'|'subscription'|'generic';
export type Concept = 'pricing'|'lifecycle'|'support'|'engagement'|'contract'|'satisfaction'|'loyalty';
export interface ColumnProfile {name:string;inferred_type:string;missing:number;unique:number;sample_values:any[];identifier?:boolean;sensitive?:boolean;leakage_warning?:string; suggested_role?:string; labels?:string[]; [key:string]:any}
export interface Feature {column:string;kind:FeatureKind;ordinal_order?:string[];date_reference?:string}
export interface Semantic {column:string;concept:Concept;meaning:string;confirmed:boolean;actionable:boolean;protected:boolean;estimated_cost?:number}
export interface Configuration {target_column:string|null;positive_label:string|null;customer_id:string|null;excluded_columns:string[];features:Feature[];business_domain:Domain;prediction_horizon:string|null;semantics:Semantic[];leakage_approved:Record<string,string>;warnings?:string[];decisions?:string[]}
export interface Profile {rows:number;columns_count:number;missing_cells:number;duplicate_rows:number;columns:ColumnProfile[];target_candidates:string[];warnings:string[];[key:string]:any}
export interface Dataset {id:string;filename:string;file_size:number;profile:Profile;configuration:Configuration|null;fingerprint:string;preview:Row[];created_at?:string}
export interface Analysis {id?:string;analysis_id:string;dataset_id:string;mode:string;status:string;created_at?:string;job_id?:string;configuration?:Configuration;model_version?:string;[key:string]:any}
export interface Job {id:string;analysis_id:string;dataset_id?:string;status:'queued'|'running'|'completed'|'failed';progress:number;message:string;error?:string|null}
export type Page = 'workspace'|'studio'|'analytics'|'models'|'explain'|'customers'|'strategy'|'reports';
export const pageNames:Record<Page,string>={workspace:'Workspace',studio:'Dataset Studio',analytics:'Executive Analytics',models:'Model Lab',explain:'Explainability',customers:'Customer Intelligence',strategy:'Retention Strategy',reports:'Reports & Methodology'};
