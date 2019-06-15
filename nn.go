/* This package implements a WANN (weight agnostig neural network) */

package main

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"math"
)

type Node interface {
	Value(edgeWeight float64) float64
}

type ConstantNode struct {
	value float64
}

func (c ConstantNode) String() string {
	return fmt.Sprintf(`Cn{%f}`, c.value)
}

func (c *ConstantNode) Value(edgeWeight float64) float64 {
	/* Edge weight is ignored here because we don't pass another edge to get this nodes' value */
	return c.value
}

type TanhNode struct {
	inputs []Node
	value  float64 // last value of this node
}

func (t TanhNode) String() string {
	return fmt.Sprintf(`Tn{%f}`, t.value)
}

func (t *TanhNode) Value(edgeWeight float64) float64 {
	sum := float64(0)

	for _, input := range t.inputs {
		sum += input.Value(edgeWeight) * edgeWeight
	}

	t.value = math.Tanh(sum)
	return t.value
}

type Edge struct {
	from Node
	to   Node
}

func (e Edge) String() string {
	return fmt.Sprintf(`E{F:%v, T:%v}`, e.from, e.to)
}

type Network struct {
	edges   []Edge
	outputs []Node
	inputs  []Node
}

func NewNetwork() Network {
	input1 := ConstantNode{}
	input2 := ConstantNode{}
	output := TanhNode{
		inputs: []Node{&input1, &input2},
	}

	edge1 := Edge{from: &input1, to: &output}
	edge2 := Edge{from: &input2, to: &output}

	return Network{
		edges:   []Edge{edge1, edge2},
		outputs: []Node{&output},
		inputs:  []Node{&input1, &input2},
	}
}

func (n Network) String() string {
	return fmt.Sprintf("N{\n\te{%v},\n\ti{%v},\n\to{%v}}", n.edges, n.inputs, n.outputs)
}

func (n *Network) feed(input []float64, edgeWeight float64) ([]float64, error) {
	if len(input) != len(n.inputs) {
		return nil, errors.New(`invalid input size`)
	}

	for idx, inputNode := range n.inputs {
		inputNode := inputNode.(*ConstantNode)
		inputNode.value = input[idx]
	}

	output := []float64{}
	for _, outputNode := range n.outputs {
		output = append(output, outputNode.Value(edgeWeight))
	}

	return output, nil
}

func (n *Network) addRandomEdge() {
	/* Adds a new edge between existing nodes */

	/* Candidates for from-nodes are inputs and inner edge nodes, candidates for to-nodes are outputs and inner edge nodes */
	toNodesSet := make(map[Node]bool)
	fromNodesSet := make(map[Node]bool)
	for _, input := range n.inputs {
		fromNodesSet[input] = true
	}
	for _, output := range n.outputs {
		toNodesSet[output] = true
	}
	for _, edge := range n.edges {
		fromNodesSet[edge.from] = true
		toNodesSet[edge.to] = true
	}

	toNodes := make([]Node, 0, len(toNodesSet))
	for node, _ := range toNodesSet {
		toNodes = append(toNodes, node)
	}
	fromNodes := make([]Node, 0, len(fromNodesSet))
	for node, _ := range fromNodesSet {
		fromNodes = append(fromNodes, node)
	}

	/* Select random to and from node */
	fromIdx := rand.Intn(len(fromNodes))
	fromNode := fromNodes[fromIdx]
	var toNode Node
	for {
		toIdx := rand.Intn(len(toNodes))
		toNode = toNodes[toIdx]
		if fromNode != toNode {
			break
		}
	}

	/* Build new edge */
	newEdge := Edge{
		from: fromNode,
		to: toNode,
	}

	n.edges = append(n.edges, newEdge)

	/* Hook up source to destination */
	toNode.(*TanhNode).inputs = append(toNode.(*TanhNode).inputs, fromNode)
}

func (n *Network) splitRandomEdge() {
	/* Selects a random edge and splits it, inserting a Tanh node in between */
	splitIdx := rand.Intn(len(n.edges))

	splitEdge := n.edges[splitIdx]

	newNode := TanhNode{
		inputs: []Node{splitEdge.from},
	}

	/* Create new edges between from and to, going over newNode */
	newEdge1 := Edge{
		from: splitEdge.from,
		to: &newNode,
	}
	newEdge2 := Edge{
		from: &newNode,
		to: splitEdge.to,
	}

	/* Remove splitEdge.from from splitEdge.to's input node list and add new split Node */
	newInputs := []Node{}
	for _, input := range splitEdge.to.(*TanhNode).inputs { /* TODO: Remove cast */
		if input != splitEdge.from {
			newInputs = append(newInputs, input)
		}
	}
	newInputs = append(newInputs, &newNode)
	splitEdge.to.(*TanhNode).inputs = newInputs /* TODO: Remove cast */

	/* Remove split edge */
	newEdges := []Edge{}
	for _, edge := range n.edges {
		if edge != splitEdge {
			newEdges = append(newEdges, edge)
		}
	}
	/* Add new edges */
	newEdges = append(newEdges, newEdge1)
	newEdges = append(newEdges, newEdge2)
	n.edges = newEdges
}

func main() {
	input := []float64{0.5, -0.5}

	net := NewNetwork()

	out, err := net.feed(input, 1.0)
	if err != nil {
		log.Fatalln(`can't feed`, err)
	}
	log.Println(`net before split:`, net, `output`, out)

	net.splitRandomEdge()
	net.addRandomEdge()

	out, err = net.feed(input, 1.0)
	if err != nil {
		log.Fatalln(`can't feed`, err)
	}
	log.Println(`net after split:`, net, `output`, out)
}
