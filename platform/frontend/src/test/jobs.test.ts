import {describe,it,expect} from 'vitest';
import {jobBelongs,recoverableJob} from '../features/jobs';
import type {Analysis,Job} from '../types';
const analysis:Analysis={analysis_id:'a',dataset_id:'dataset-a',status:'running',mode:'historical',job_id:'job-a'};
const job:Job={id:'job-a',analysis_id:'a',dataset_id:'dataset-a',status:'running',progress:25,message:'Actual job progress'};
describe('analysis-scoped job state',()=>{
 it('recovers the selected analysis active job',()=>{expect(recoverableJob(analysis)).toBe('job-a');expect(jobBelongs(job,'a','dataset-a')).toBe(true)});
 it('does not display another dataset or analysis job',()=>{expect(jobBelongs(job,'b','dataset-a')).toBe(false);expect(jobBelongs(job,'a','dataset-b')).toBe(false);expect(jobBelongs(undefined,'a','dataset-a')).toBe(false)});
 it('does not resurrect completed progress after selection clears it',()=>{expect(recoverableJob({...analysis,status:'completed'})).toBe('');expect(recoverableJob(undefined)).toBe('');expect(recoverableJob({...analysis,status:'failed'})).toBe('job-a')});
});
