# activation
--
    import "."


## Usage

#### type Activation

```go
type Activation interface {
	Forward(float64) float64
	Backward(float64) float64
}
```

Activation represents an activation function.

#### type ELU

```go
type ELU struct {
	A float64
}
```

ELU is an Exponential Linear Unit activation. It computes max(A * (e^x - 1), x).

#### func (ELU) Backward

```go
func (e ELU) Backward(x float64) float64
```

#### func (ELU) Forward

```go
func (e ELU) Forward(x float64) float64
```

#### type Gaussian

```go
type Gaussian struct{}
```


#### func (Gaussian) Backward

```go
func (g Gaussian) Backward(x float64) float64
```

#### func (Gaussian) Forward

```go
func (g Gaussian) Forward(x float64) float64
```

#### type LeakyReLU

```go
type LeakyReLU struct {
	Leak float64
	Cap  float64
}
```

LeakyReLU is a Leaky Rectified Linear Unit activation with activation cap.

If Cap is 0, the unit is uncapped. Otherwise, the output is clipped between -Cap
and +Cap

#### func (LeakyReLU) Backward

```go
func (r LeakyReLU) Backward(x float64) float64
```

#### func (LeakyReLU) Forward

```go
func (r LeakyReLU) Forward(x float64) float64
```

#### type Sigmoid

```go
type Sigmoid struct{}
```

Sigmoid is a sigmoid activation function. It computes 1/(1 + e^(-x)).

#### func (Sigmoid) Backward

```go
func (s Sigmoid) Backward(x float64) float64
```

#### func (Sigmoid) Forward

```go
func (s Sigmoid) Forward(x float64) float64
```

#### type Softplus

```go
type Softplus struct{}
```


#### func (Softplus) Backward

```go
func (s Softplus) Backward(v float64) float64
```

#### func (Softplus) Forward

```go
func (s Softplus) Forward(v float64) float64
```

#### type Tanh

```go
type Tanh struct{}
```

Tanh computes tanh(x) as the activation function.

#### func (Tanh) Backward

```go
func (t Tanh) Backward(x float64) float64
```

#### func (Tanh) Forward

```go
func (t Tanh) Forward(x float64) float64
```
