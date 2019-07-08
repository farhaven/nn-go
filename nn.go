package main

import (
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"
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
	// logger := log.New(os.Stdout, `[MAIN ] `, log.LstdFlags)

	rand.Seed(time.Now().Unix())

	/*
	samples := ReadMnist(`train`)
	logger.Println(`training data loaded, starting training`)

	network := NewNetwork([]int{28 * 28, 300, 10})
	*/
	samples := []Sample{
		Sample{[]float64{0, 0}, []float64{0}},
		Sample{[]float64{0, 1}, []float64{1}},
		Sample{[]float64{1, 0}, []float64{1}},
		Sample{[]float64{1, 1}, []float64{0}},
	}
	network := NewNetwork([]int{2, 3, 1})

	go profTask()

	trainNetwork(network, samples)
}
