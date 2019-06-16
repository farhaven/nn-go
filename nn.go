package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().Unix())
	log.Println(`here we go`)

	samples := []Sample{}
	for idx := 0; idx < 30; idx++ {
		in := float64(idx) * ((math.Pi * 2) / 30)
		t1 := math.Cos(in)
		t2 := math.Sin(in)
		t3 := math.Tanh(in)

		samples = append(samples, Sample{[]float64{in}, []float64{t1, t2, t3}})
	}

	networks := []*Network{}
	for idx := 0; idx < epochSlice*2; idx++ {
		networks = append(networks, NewNetwork(1, 3))
	}

	networks = trainNetworks(networks, samples)

	for idx, net := range networks {
		net.dumpDot(fmt.Sprintf(`graphs/%03d.dot`, idx), samples)
	}

	log.Println(`output of best network:`)
	for _, s := range samples {
		networks[0].feed(s.inputs)
		log.Println(`in:`, s.inputs, `out:`, networks[0].getOutput(), `target:`, s.targets)
	}
}
