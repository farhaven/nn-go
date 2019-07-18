package network

import "math"

// Activation represents an activation function.
type Activation interface {
	Forward(float64) float64
	Backward(float64) float64
}

// TanhActivation computes tanh(x) as the activation function.
type TanhActivation struct{}

func (t TanhActivation) Forward(x float64) float64 {
	return math.Tanh(x)
}
func (t TanhActivation) Backward(x float64) float64 {
	return 1 - math.Pow(x, 2.0)
}

// ELUActivation is an Exponential Linear Unit activation. It computes max(A * (e^x - 1), x).
type ELUActivation struct {
	A float64
}

func (e ELUActivation) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return e.A * (math.Exp(x) - 1)
}

func (e ELUActivation) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return e.A * math.Exp(x)
}

// LeakyRELUActivation is a Leaky Rectified Linear Unit activation with activation cap.
//
// If Cap is 0, the unit is uncapped. Otherwise, the output is clipped between -Cap and +Cap
type LeakyRELUActivation struct {
	Leak float64
	Cap  float64
}

func (r LeakyRELUActivation) Forward(x float64) float64 {
	res := x
	if x < 0 {
		res *= r.Leak
	}

	if r.Cap == 0 {
		return res
	}

	return math.Max(-r.Cap, math.Min(r.Cap, res))
}
func (r LeakyRELUActivation) Backward(x float64) float64 {
	if x < 0 {
		return r.Leak
	}
	return 1.0
}

// SigmoidActivation is a sigmoid activation function. It computes 1/(1 + e^(-x)).
type SigmoidActivation struct{}

func (s SigmoidActivation) Forward(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
func (s SigmoidActivation) Backward(x float64) float64 {
	f := s.Forward(x)
	return f * (1.0 - f)
}
