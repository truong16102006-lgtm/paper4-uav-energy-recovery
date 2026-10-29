"""Map supplied paired voltage/current error specifications to a power contract."""
import math

def power_bounds(voltage_relative_error,current_relative_error,voltage_bias_V,current_bias_A,true_voltage_upper_V,true_current_upper_A):
    values=[voltage_relative_error,current_relative_error,voltage_bias_V,current_bias_A,true_voltage_upper_V,true_current_upper_A]
    if not all(math.isfinite(float(x)) and x>=0 for x in values):raise ValueError('Finite nonnegative sensor specifications and true operating limits required.')
    gv,gi,bv,bi,V,I=map(float,values);gamma=gv+gi+gv*gi;bias=(1+gv)*V*bi+(1+gi)*I*bv+bv*bi
    if gamma>=1:raise ValueError('Derived relative power uncertainty must be below one for this finite energy enclosure.')
    return dict(relative_power_gain_bound=gamma,additive_power_bias_bound_W=bias,assumptions='Paired synchronous nonnegative true voltage/current within supplied upper limits; affine bounded component errors',physically_validated_by_this_code=False)
