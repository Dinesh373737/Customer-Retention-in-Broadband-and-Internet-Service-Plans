import {describe,it,expect} from 'vitest';
import {initialColumns,buildConfiguration,labelString} from '../features/schema';
import type {Profile} from '../types';
const fixture:Profile={rows:100,columns_count:5,missing_cells:1,duplicate_rows:0,target_candidates:['Exited'],warnings:[],columns:[
 {name:'CustomerId',inferred_type:'numeric',missing:0,unique:100,sample_values:[101,102],identifier:true},
 {name:'Fee',inferred_type:'numeric',missing:1,unique:50,sample_values:[12,20]},
 {name:'Age',inferred_type:'numeric',missing:0,unique:40,sample_values:[20,50],sensitive:true,protected:true},
 {name:'Exited',inferred_type:'binary',missing:0,unique:2,sample_values:[0,1],target_labels:['0','1']},
 {name:'OutcomeReason',inferred_type:'categorical',missing:0,unique:3,sample_values:['a','b'],leakage_warning:'Post-outcome risk'},
]};
describe('explicit user-confirmed schema mapping (test fixtures)',()=>{
 it('excludes identifiers, protected fields, likely targets and leakage suggestions by default',()=>{
  const columns=initialColumns(fixture);expect(columns.CustomerId.include).toBe(false);expect(columns.Age.include).toBe(false);expect(columns.Exited.include).toBe(false);expect(columns.OutcomeReason.include).toBe(false);expect(columns.Fee.include).toBe(true);
  const config=buildConfiguration(fixture,columns,'Exited','1','CustomerId','banking','');expect(config.features.map(f=>f.column)).toEqual(['Fee']);expect(config.positive_label).toBe('1');expect(config.customer_id).toBe('CustomerId');expect(config.semantics).toEqual([]);expect(config.prediction_horizon).toBeNull();
 });
 it('does not fabricate a target in exploratory mode',()=>{
  const config=buildConfiguration(fixture,initialColumns(fixture),'','','CustomerId','generic','');expect(config.target_column).toBeNull();expect(config.positive_label).toBeNull();
 });
 it('requires explicit ordinal order and a fixed date reference',()=>{
  const c=initialColumns(fixture);c.Fee.kind='ordinal';expect(()=>buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','')).toThrow('ordered levels');c.Fee.ordinal='12,20';expect(buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','').features[0].ordinal_order).toEqual(['12','20']);c.Fee.kind='datetime';expect(()=>buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','')).toThrow('reference date');
 });
 it('keeps uncertain semantics unconfirmed and rejects protected actionable changes',()=>{
  const c=initialColumns(fixture);c.Fee.concept='pricing';c.Fee.meaning='Recorded subscription fee';const config=buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','');expect(config.semantics[0].confirmed).toBe(false);expect(config.semantics[0].actionable).toBe(false);c.Fee.protected=true;c.Fee.actionable=true;expect(()=>buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','')).toThrow('cannot be actionable');
 });
 it('requires a positive class and meaningful confirmation',()=>{
  const c=initialColumns(fixture);expect(()=>buildConfiguration(fixture,c,'Exited','','CustomerId','banking','')).toThrow('label');c.Fee.concept='pricing';c.Fee.confirmed=true;c.Fee.meaning='';expect(()=>buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','')).toThrow('business meaning');
 });
 it('allows descriptive historical analysis when no eligible predictor remains',()=>{const c=initialColumns(fixture);c.Fee.include=false;const config=buildConfiguration(fixture,c,'Exited','1','CustomerId','banking','');expect(config.features).toEqual([]);expect(config.target_column).toBe('Exited');expect(config.semantics).toEqual([])});
 it('keeps valid zero and boolean label encodings visible',()=>{expect(labelString(0)).toBe('0');expect(labelString(false)).toBe('false');expect(labelString(true)).toBe('true')});
});
