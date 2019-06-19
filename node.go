package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Node interface {
	getValue() float64
	updateValue()
	clone() Node
}

type ConstantNode struct {
	value float64
}

func (c *ConstantNode) String() string {
	return fmt.Sprintf(`C:%p:%f`, c, c.value)
}

func (c *ConstantNode) getValue() float64 {
	return c.value
}

func (c *ConstantNode) updateValue() {
	/* Do nothing here, the value does not depend on any other nodes */
}

func (c *ConstantNode) clone() Node {
	return &ConstantNode{c.value}
}

type Operation interface {
	Apply([]float64) float64
	Id() int
}

func sum(nums []float64) float64 {
	s := float64(0)
	for _, n := range nums {
		s += n
	}
	return s
}

type Invert struct{}

func (i Invert) Apply(n []float64) float64 {
	return -sum(n)
}

func (i Invert) Id() int {
	return 0
}

type Tanh struct{}

func (t Tanh) Apply(n []float64) float64 {
	return math.Tanh(sum(n))
}

func (t Tanh) Id() int {
	return 1
}

type ELU struct{}

func (r ELU) Apply(n []float64) float64 {
	s := sum(n)

	if s < 0 {
		return -math.Log(-s)
	}

	return s
}

func (r ELU) Id() int {
	return 2
}

type Sine struct{}

func (s Sine) Apply(n []float64) float64 {
	return math.Sin(sum(n))
}

func (s Sine) Id() int {
	return 3
}

type Identity struct{}

func (i Identity) Apply(n []float64) float64 {
	return sum(n)
}
func (i Identity) Id() int {
	return 4
}

type Gauss struct{}

func (g Gauss) Apply(n []float64) float64 {
	sigma := 1.0
	mu := 0.0

	z := rand.Float64()*2 - 1.0

	r := (1.0 / (sigma * math.Sqrt(2*math.Pi))) * math.Exp(-math.Pow(z-mu, 2)/(2*math.Pow(sigma, 2)))

	return sum(n) + r
}

func (g Gauss) Id() int {
	return 5
}

type MaxPool struct {}

func (m MaxPool) Apply(nums []float64) float64 {
	max := math.Inf(-1)
	for _, num := range nums {
		if num > max {
			max = num
		}
	}
	return max
}

func (m MaxPool) Id() int {
	return 6
}

func RandomOperation() Operation {
	switch rand.Intn(6) {
	case 0:
		return Invert{}
	case 1:
		return Tanh{}
	case 2:
		return ELU{}
	case 3:
		return Sine{}
	case 4:
		return Identity{}
	case 5:
		return Gauss{}
	case 6:
		return MaxPool{}
	default:
		panic(`unknown operation`)
	}
}

type SumNode struct {
	op     Operation
	inputs []Node
	value  float64
	inputScratch []float64
}

func NewSumNode(inputs []Node) *SumNode {
	return &SumNode{
		op:     RandomOperation(),
		inputs: inputs,
		inputScratch: make([]float64, len(inputs)), /* Lazy realloc'ed scratch space to save on allocations */
	}
}

func (s *SumNode) clone() Node {
	return &SumNode{
		op: s.op,
	}
}

func (s *SumNode) graphViz() string {
	label := fmt.Sprintf("%f", s.value)
	style := ", fontname=\"Ubuntu Mono\""

	switch s.op.(type) {
	case Invert:
		style += ", style=filled, fillcolor=red"
		label = "± " + label
	case Tanh:
		style += ", style=filled, fillcolor=gray"
		label = "∫ " + label
	case Sine:
		label = "∿ " + label
	case ELU:
		label = "⦧ " + label
	case Identity:
		label = "⦿ " + label
	case Gauss:
		style += ", style=filled, fillcolor=lightblue"
		label = "⦿ " + label
	case MaxPool:
		style += ", style=filled, fillcolor=lightblue"
		label = "⩓ " + label
	default:
		panic(`unknown node operation`)
	}

	res := fmt.Sprintf("\t\"%p\" [label=\"%s\"", s, label)
	res += style + "]\n"
	return res
}

func (s *SumNode) updateValue() {
	if len(s.inputs) != len(s.inputScratch) {
		s.inputScratch = make([]float64, len(s.inputs))
	}

	for idx, input := range s.inputs {
		s.inputScratch[idx] = input.getValue()
	}

	s.value = s.op.Apply(s.inputScratch)
}

func (s *SumNode) getValue() float64 {
	return s.value
}
