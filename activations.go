package network

import "math"

type Activation interface {
	Forward(float64) float64
	Backward(float64) float64
}

type TanhActivation struct{}

func (t TanhActivation) Forward(x float64) float64 {
	return math.Tanh(x)
}
func (t TanhActivation) Backward(x float64) float64 {
	return 1 - math.Pow(x, 2.0)
}

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

type LeakyRELUActivation struct {
	Leak float64
	Cap  float64
}

func (r LeakyRELUActivation) Forward(x float64) float64 {
	if x >= 0 {
		if r.Cap > 0 {
			return math.Min(x, r.Cap)
		}
		return x
	}
	return x * r.Leak
}
func (r LeakyRELUActivation) Backward(x float64) float64 {
	if x < 0 {
		return r.Leak
	}
	return 1.0
}

type SigmoidActivation struct{}

func (s SigmoidActivation) Forward(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
func (s SigmoidActivation) Backward(x float64) float64 {
	f := s.Forward(x)
	return f * (1.0 - f)
}
