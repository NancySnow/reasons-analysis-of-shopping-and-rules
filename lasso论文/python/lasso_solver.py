"""Convex mean logistic + lambda L1 with unpenalized intercept, explicit KKT gate."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy import sparse

def kkt(X,y,w,b,lam):
 p=expit(X@w+b);res=p-y;g=np.asarray(X.T@res).ravel()/len(y);gi=float(np.mean(res));nz=np.abs(w)>1e-10
 rr=np.maximum(np.abs(g)-lam,0);rr[nz]=np.abs(g[nz]+lam*np.sign(w[nz]))
 return {'intercept_abs_gradient':abs(gi),'coefficient_max_KKT':float(rr.max(initial=0)),'max_KKT':max(abs(gi),float(rr.max(initial=0))),'mean_train_prediction':float(p.mean()),'train_positive_rate':float(np.mean(y)),'objective':float(np.mean(np.logaddexp(0,X@w+b)-y*(X@w+b))+lam*np.abs(w).sum())}
class CheckedModel:
 def __init__(self,w,b,nit,diagnostic):self.coef_=w.reshape(1,-1);self.intercept_=np.array([b]);self.classes_=np.array([0,1]);self.n_iter_=np.array([nit]);self.diagnostic=diagnostic
 def predict_proba(self,X):
  p=expit(X@self.coef_[0]+self.intercept_[0]);return np.column_stack((1-p,p))
def fit_checked(X,y,lam,*,tolerance=1e-7,max_iter=20000):
 y=np.asarray(y,float);n,d=X.shape
 if set(y)!={0.,1.}:raise ValueError('Two fitting classes required')
 prior=float(y.mean());b0=float(np.log(prior/(1-prior)));w0=np.zeros(d);z=kkt(X,y,w0,b0,lam)
 if z['max_KKT']<=tolerance:
  z.update(solver='exact_intercept_only_MLE_with_full_L1_KKT_check',iterations=0,converged=True);return CheckedModel(w0,b0,0,z)
 # Split w=u-v >=0; this is the same convex L1 objective, not an approximation.
 def fun(theta):
  w=theta[:d]-theta[d:2*d];b=theta[-1];eta=X@w+b;p=expit(eta);res=p-y;g=np.asarray(X.T@res).ravel()/n
  f=np.mean(np.logaddexp(0,eta)-y*eta)+lam*np.sum(theta[:2*d]);return float(f),np.r_[g+lam,-g+lam,np.mean(res)]
 theta=np.r_[np.zeros(2*d),b0];total=0;attempts=[]
 for restart in range(3):
  result=minimize(fun,theta,jac=True,method='L-BFGS-B',bounds=[(0,None)]*(2*d)+[(None,None)],options={'ftol':1e-15,'gtol':1e-9,'maxiter':max_iter,'maxls':50,'maxcor':10})
  theta=result.x;total+=int(result.nit);w=theta[:d]-theta[d:2*d];b=theta[-1];z=kkt(X,y,w,b,lam);attempts.append({'status':int(result.status),'message':str(result.message),'iterations':int(result.nit),'max_KKT':z['max_KKT']})
  if z['max_KKT']<=tolerance:
   z.update(solver='L-BFGS-B_nonnegative_split_exact_L1',iterations=total,converged=True,attempts=attempts);return CheckedModel(w,b,total,z)
 raise RuntimeError('KKT convergence gate not met: '+repr(attempts))
