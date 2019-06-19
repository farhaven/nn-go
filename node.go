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
	Apply(float64) float64
	Id() int
}

type Invert struct{}

func (i Invert) Apply(n float64) float64 {
	return -n
}

func (i Invert) Id() int {
	return 0
}

type Tanh struct{}

func (t Tanh) Apply(n float64) float64 {
	return math.Tanh(n)
}

func (t Tanh) Id() int {
	return 1
}

type ELU struct{}

func (r ELU) Apply(n float64) float64 {
	if n < 0 {
		return -math.Log(-n)
	}
	return n
}

func (r ELU) Id() int {
	return 2
}

type Sine struct{}

func (s Sine) Apply(n float64) float64 {
	return math.Sin(n)
}

func (s Sine) Id() int {
	return 3
}

type Identity struct{}

func (i Identity) Apply(n float64) float64 {
	return n
}
func (i Identity) Id() int {
	return 4
}

type Gauss struct{}

func (g Gauss) Apply(n float64) float64 {
	sigma := 1.0
	mu := 0.0

	z := rand.Float64()*2 - 1.0

	r := (1.0 / (sigma * math.Sqrt(2*math.Pi))) * math.Exp(-math.Pow(z-mu, 2)/(2*math.Pow(sigma, 2)))

	return n + r
}

func (g Gauss) Id() int {
	return 5
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
	default:
		panic(`unknown operation`)
	}
}

type SumNode struct {
	op     Operation
	inputs []Node
	value  float64
}

func NewSumNode(inputs []Node) *SumNode {
	return &SumNode{
		op:     RandomOperation(),
		inputs: inputs,
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

	if s.op == (Invert{}) {
		style += ", style=filled, fillcolor=red"
		label = "± " + label
	} else if s.op == (Tanh{}) {
		style += ", style=filled, fillcolor=gray"
		label = "∫ " + label
	} else if s.op == (Sine{}) {
		label = "∿ " + label
	} else if s.op == (ELU{}) {
		label = "⦧ " + label
	} else if s.op == (Identity{}) {
		label = "⦿ " + label
	} else if s.op == (Gauss{}) {
		style += ", style=filled, fillcolor=lightblue"
		label = "⦿ " + label
	}

	res := fmt.Sprintf("\t\"%p\" [label=\"%s\"", s, label)
	res += style + "]\n"
	return res
}

func (s *SumNode) updateValue() {
	sum := float64(0)

	for _, input := range s.inputs {
		sum += input.getValue()
	}

	s.value = s.op.Apply(sum)
}

func (s *SumNode) getValue() float64 {
	return s.value
}
