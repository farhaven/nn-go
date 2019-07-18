package main

import (
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	network "github.com/farhaven/nn-go"
)

func profTask() {
	logger := log.New(os.Stdout, `[PROF ] `, log.LstdFlags)
	proffd, err := os.Create("cpuprofile.pprof")
	if err != nil {
		logger.Fatalln(`can't create CPU profile:`, err)
	}

	pprof.StartCPUProfile(proffd)
	logger.Println(`started profiling`)
	defer func() {
		pprof.StopCPUProfile()
		logger.Println(`stopped profiling`)
	}()

	select {
	case <-time.After(120 * time.Second):
		logger.Println(`profile timer expired`)
	}
}

func main() {
	logger := log.New(os.Stdout, `[MAIN ] `, log.LstdFlags)

	rand.Seed(time.Now().Unix())

	samples := readMnist(`train`)
	logger.Println(`training data loaded, starting training`)

	config := []network.LayerConfiguration{
		network.LayerConfiguration{28 * 28, nil},
		network.LayerConfiguration{800, network.LeakyRELUActivation{Leak: 0.001, Cap: 1e6}},
		network.LayerConfiguration{10, network.SigmoidActivation{}},
	}
	net, err := network.NewNetwork(config)
	if err != nil {
		log.Fatalln(`can't create network:`, err)
	}

	go profTask()

	trainNetwork(net, samples)

	// Evaluate network on the test set
	logger.Println(`evaluating network on test set`)
	errors := 0
	samples = readMnist(`t10k`)
	for _, s := range samples {
		output := net.Forward(s.input)
		label := maxIdx(output)
		if label != maxIdx(s.target) {
			errors += 1
		}
	}
	errorRate := float64(errors) / float64(len(samples) + 1)
	logger.Printf(`errors: %d/%d (%.3f%% error)`, errors, len(samples), errorRate * 100)
}
