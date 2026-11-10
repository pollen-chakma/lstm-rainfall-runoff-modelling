import numpy as np
from netCDF4 import num2date
from prepare_daymet import numeric

def dates(ds,observation=False):
 t=ds['time']; b=ds['time_bnds']; bounds=numeric(b)
 if b.dimensions[0]!='time': bounds=bounds.T
 units=getattr(b,'units',t.units); calendar=getattr(b,'calendar',getattr(t,'calendar','standard'))
 assert calendar in ['standard','gregorian','proleptic_gregorian'],calendar
 unit=units.split(' since ')[0].lower().strip()
 factors={'seconds':1,'second':1,'minutes':60,'minute':60,'hours':3600,'hour':3600,'days':86400,'day':86400}
 factor=factors[unit]
 assert np.isfinite(bounds).all()
 assert np.allclose((bounds[:,1]-bounds[:,0])*factor,86400,rtol=0,atol=1e-4),'Non-daily bounds'
 first=num2date(bounds[0,0],units,calendar=calendar)
 origin=np.datetime64(first.strftime('%Y-%m-%dT%H:%M:%S'),'us')
 delta=np.rint((bounds[:,0]-bounds[0,0])*factor*1e6).astype('timedelta64[us]')
 starts=origin+delta
 days=starts.astype('datetime64[D]')
 assert (starts==days.astype('datetime64[us]')).all(),'Bounds not midnight aligned'
 assert (np.diff(days).astype('timedelta64[D]').astype(int)>0).all(),'Duplicate/nonmonotonic dates'
 return days
