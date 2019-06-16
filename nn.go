/* This package implements a WANN (weight agnostig neural network) */

package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
)

type Node interface {
	Value(edgeWeight float64) float64
	Clone() Node
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

func (c *ConstantNode) Clone() Node {
	return &ConstantNode{c.value}
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

func (t *TanhNode) Clone() Node {
	return &TanhNode{
		inputs: []Node{},
		value:  t.value,
	}
}

type Edge struct {
	from Node
	to   Node
}

func (e Edge) String() string {
	return fmt.Sprintf(`E{F:%v, T:%v}`, e.from, e.to)
}

type Network struct {
	edges      []Edge
	outputs    []Node
	inputs     []Node
	totalError float64
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

func (n *Network) Clone() Network {
	/* Create clones of all nodes, then reconstruct edges and input mappings */
	clones := make(map[Node]Node)
	for _, edge := range n.edges {
		clones[edge.from] = edge.from.Clone()
		clones[edge.to] = edge.to.Clone()
	}

	/* Rebuild edges and input mappings */
	newEdges := []Edge{}
	for _, edge := range n.edges {
		e := Edge{
			from: clones[edge.from],
			to:   clones[edge.to],
		}
		newEdges = append(newEdges, e)

		/* Rebuild input nodes if the edge.to is a TanhNode */
		if th, ok := edge.to.(*TanhNode); ok {
			clone := clones[edge.to].(*TanhNode)
			for _, input := range th.inputs {
				clone.inputs = append(clone.inputs, clones[input])
			}
		}
	}

	/* Gather new input and output nodes */
	newInputs := []Node{}
	for _, input := range n.inputs {
		newInputs = append(newInputs, clones[input])
	}
	newOutputs := []Node{}
	for _, output := range n.outputs {
		newOutputs = append(newOutputs, clones[output])
	}

	return Network{
		edges:   newEdges,
		inputs:  newInputs,
		outputs: newOutputs,
	}
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
		to:   toNode,
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
		to:   &newNode,
	}
	newEdge2 := Edge{
		from: &newNode,
		to:   splitEdge.to,
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

type Sample struct {
	inputs  []float64
	targets []float64
}

/* trainingError returns the squared distance of the network output from the desired sample output */
func (n *Network) trainingError(sample Sample, edgeWeight float64) (float64, error) {
	if len(sample.targets) != len(n.outputs) {
		return 0, errors.New(`invalid target size`)
	}

	outputs, err := n.feed(sample.inputs, edgeWeight)
	if err != nil {
		return 0, err
	}

	sum := float64(0)

	for idx, output := range outputs {
		sum += math.Pow(output-sample.targets[idx], 2)
	}

	return sum, nil
}

/* totalError returns the sum of the total error for all training examples */
func (n *Network) updateTotalError(samples []Sample, edgeWeight float64) error {
	sum := float64(0)

	for _, s := range samples {
		te, err := n.trainingError(s, edgeWeight)
		if err != nil {
			return err
		}
		sum += te
	}

	n.totalError = sum
	return nil
}

const numEpochs = 5
const splitAdd = 10

func main() {
	samples := []Sample{
		Sample{[]float64{-1, -1}, []float64{-1}},
		Sample{[]float64{-1, 1}, []float64{1}},
		Sample{[]float64{1, -1}, []float64{1}},
		Sample{[]float64{1, 1}, []float64{-1}},
	}

	networks := []*Network{}
	for idx := 0; idx < 10; idx++ {
		net := NewNetwork()
		if rand.Intn(splitAdd) == 0 {
			net.addRandomEdge()
		} else {
			net.splitRandomEdge()
		}
		networks = append(networks, &net)
	}

	for epoch := 0; epoch < numEpochs; epoch++ {
		for _, net := range networks {
			err := net.updateTotalError(samples, 1.0)
			if err != nil {
				log.Fatalln(`can't get total error:`, err)
			}
		}

		sort.Slice(networks, func(i, j int) bool {
			return networks[i].totalError < networks[j].totalError
		})

		/* Cut off lower half of population */
		networks = networks[:len(networks)/2]

		/* Clone and mutate upper half of population */
		clones := []*Network{}
		for _, net := range networks {
			clone := net.Clone()
			if rand.Intn(splitAdd) == 0 {
				net.addRandomEdge()
			} else {
				net.splitRandomEdge()
			}
			clones = append(clones, &clone)
		}
		networks = append(networks, clones...)
	}

	log.Println(`errors after`, numEpochs, `epochs:`)
	for _, net := range networks {
		if err := net.updateTotalError(samples, 1.0); err != nil {
			log.Fatalln(`can't update total error`, err)
		}
	}
	sort.Slice(networks, func(i, j int) bool {
		return networks[i].totalError < networks[j].totalError
	})
	for idx, net := range networks {
		log.Println(idx, `->`, net.totalError)
	}

	log.Println(`output of best network:`)
	for _, sample := range samples {
		outputs, err := networks[0].feed(sample.inputs, 1.0)
		if err != nil {
			log.Fatalln(`can't get output from network:`, err)
		}
		log.Println(`in:`, sample.inputs, `out:`, outputs, `target:`, sample.targets)
	}
}
