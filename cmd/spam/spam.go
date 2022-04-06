package main

import (
	"flag"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	network "github.com/farhaven/nn-go"
	"github.com/farhaven/nn-go/activation"
)

type trainAs int

const (
	TrainHam = trainAs(iota)
	TrainSpam
	TrainNone
)

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

const (
	ngramSize  = 16
	memorySize = 2
)

func train(r io.Reader, net *network.Network, t trainAs) error {
	n := NGram{
		r: r,
		n: ngramSize,
	}

	var (
		input  [ngramSize + memorySize]float64
		target []float64
	)

	score := make([]float64, memorySize)

	switch t {
	case TrainSpam:
		target = []float64{1, 0}
	case TrainHam:
		target = []float64{0, 1}
	}

	for {
		rawInput, err := n.Scan()
		if err != nil {
			break
		}

		for i, r := range rawInput {
			input[i] = float64(r)
		}

		// Old score
		input[ngramSize] = score[0]
		input[ngramSize+1] = score[1]

		p := net.Forward(input[:])

		if t != TrainNone {
			net.Backprop(input[:], network.Error(p, target), 0.01)
		}

		score[0] = p[0]
		score[1] = p[1]
	}

	log.Println("final score:", score)

	if math.Abs(score[0]-score[1]) > 0.5 {
		if score[0] > score[1] {
			log.Println("looks like spam")
		} else {
			log.Println("looks like ham")
		}
	} else {
		log.Println("dunno, bro")
	}

	return nil
}

func main() {
	input := flag.String("input", "-", "input. if -, reads from stdin")
	class := flag.String("class", "none", "spam or ham")
	name := flag.String("name", "/tmp/brain", "name for persisting the network")

	flag.Parse()

	rand.Seed(time.Now().UnixNano())

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
		{Inputs: ngramSize + memorySize},
		{Inputs: 40, Activation: activation.Tanh{}},
		{Inputs: 2, Activation: activation.Tanh{}},
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
