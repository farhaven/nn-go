package activation

import "math"

// Activation represents an activation function.
type Activation interface {
	Forward(float64) float64
	Backward(float64) float64
}

// Tanh computes tanh(x) as the activation function.
type Tanh struct{}

func (t Tanh) Forward(x float64) float64 {
	return math.Tanh(x)
}
func (t Tanh) Backward(x float64) float64 {
	return 1 - math.Pow(x, 2.0)
}

// ELU is an Exponential Linear Unit activation. It computes max(A * (e^x - 1), x).
type ELU struct {
	A float64
}

func (e ELU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return e.A * (math.Exp(x) - 1)
}

func (e ELU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return e.A * math.Exp(x)
}

// LeakyReLU is a Leaky Rectified Linear Unit activation with activation cap.
//
// If Cap is 0, the unit is uncapped. Otherwise, the output is clipped between -Cap and +Cap
type LeakyReLU struct {
	Leak float64
	Cap  float64
}

func (r LeakyReLU) Forward(x float64) float64 {
	res := x
	if x < 0 {
		res *= r.Leak
	}

	if r.Cap == 0 {
		return res
	}

	return math.Max(-r.Cap, math.Min(r.Cap, res))
}
func (r LeakyReLU) Backward(x float64) float64 {
	if x < 0 {
		return r.Leak
	}
	return 1.0
}

// Sigmoid is a sigmoid activation function. It computes 1/(1 + e^(-x)).
type Sigmoid struct{}

func (s Sigmoid) Forward(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
func (s Sigmoid) Backward(x float64) float64 {
	f := s.Forward(x)
	return f * (1.0 - f)
}

type Softplus struct{}

func (s Softplus) Forward(v float64) float64 {
	return math.Log(1 + math.Exp(v))
}

func (s Softplus) Backward(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}
