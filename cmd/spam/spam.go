package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"os"

	network "github.com/farhaven/nn-go"
	"github.com/farhaven/nn-go/activation"
)

type trainAs int

const (
	TrainHam = trainAs(iota)
	TrainSpam
	TrainNone
)

func sigmoid(x float64) float64 {
	if x < 0 || x > 1 {
		panic(fmt.Sprintf("x out of [0, 1]: %f", x))
	}

	midpoint := 0.5
	max := 1.0
	k := 5.0

	return max / (1.0 + math.Exp(-k*(x-midpoint)))
}

type NGram struct {
	r io.Reader
	n int

	buf  []byte
	rbuf [1024]byte
}

func (n *NGram) Scan() ([]byte, error) {
	if n.n > len(n.rbuf) || n.n == 0 {
		panic("ngram length must be between 1 and 1024")
	}

	if len(n.buf) < n.n {
		// Shift buffered data to the left, and read more data info the buffer after that.
		copy(n.rbuf[:], n.buf)

		// Read in more from the buffer. If we can't, return an error.
		nn, err := n.r.Read(n.rbuf[len(n.buf):])
		if err != nil {
			return nil, err
		}

		n.buf = n.rbuf[:nn+len(n.buf)]
	}

	if len(n.buf) < n.n {
		return nil, io.EOF
	}

	res := n.buf[:n.n]
	n.buf = n.buf[1:]

	return res, nil
}

func train(r io.Reader, net *network.Network, t trainAs) error {
	n := NGram{
		r: r,
		n: 16,
	}

	var input [16]float64

	target := []float64{0}
	if t == TrainSpam {
		target[0] = 1
	}

	score := big.NewFloat(1)

	for {
		rawInput, err := n.Scan()
		if err != nil {
			break
		}

		for i, r := range rawInput {
			input[i] = float64(r)
		}

		p := net.Forward(input[:])

		if t != TrainNone {
			net.Backprop(input[:], network.Error(p, target), 0.001)
		}

		score.Mul(score, big.NewFloat(sigmoid(p[0])+0.5))
	}

	log.Println("final score:", score)
	if score.Cmp(big.NewFloat(1)) <= 0 {
		log.Println("looks like ham")
	} else {
		log.Println("looks like spam")
	}

	return nil
}

func main() {
	input := flag.String("input", "-", "input. if -, reads from stdin")
	class := flag.String("class", "none", "spam or ham")
	name := flag.String("name", "/tmp/brain", "name for persisting the network")

	flag.Parse()

	var t trainAs

	switch *class {
	case "spam":
		t = TrainSpam
	case "ham":
		t = TrainHam
	case "none":
		t = TrainNone
	default:
		log.Fatalln("unknown training class:", *class)
	}

	log.Println("here we go", *input)

	config := []network.LayerConf{
		{Inputs: 16},
		{Inputs: 40, Activation: activation.LeakyReLU{Leak: 0.001, Cap: 1e6}},
		{Inputs: 40, Activation: activation.LeakyReLU{Leak: 0.001, Cap: 1e6}},
		{Inputs: 1, Activation: activation.Sigmoid{}},
	}

	net, err := network.New(config)
	if err != nil {
		log.Fatalln("can't create network:", err)
	}

	err = net.Restore(*name)
	if err != nil {
		log.Println("restoring network failed:", err)

		// Re-initialize network
		net, err = network.New(config)
		if err != nil {
			log.Fatalln("can't create network:", err)
		}
	}

	if err != nil {
		log.Fatal("loading network failed:", err)
	}

	var r io.Reader
	if *input != "-" {
		fh, err := os.Open(*input)
		if err != nil {
			log.Fatalln("opening input:", err)
		}
		defer fh.Close()

		r = fh
	} else {
		r = os.Stdin
	}

	err = train(r, net, t)
	if err != nil {
		log.Fatalln("feeding failed:", err)
	}

	err = net.Snapshot(*name)
	if err != nil {
		log.Fatalln("network persistence failed:", err)
	}
}
