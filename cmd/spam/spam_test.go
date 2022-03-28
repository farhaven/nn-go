package main

import (
	"bytes"
	"testing"
)

func TestNGram(t *testing.T) {
	buf := bytes.NewBuffer([]byte("1234567890"))

	r := NGram{
		r: buf,
		n: 3,
	}

	var prev []byte

	for {
		g, err := r.Scan()
		if err != nil {
			t.Log("err:", err)
			break
		}

		if len(g) != r.n {
			t.Error("unexpected:", g)
		}

		t.Log("ngram:", string(g))

		if prev != nil {
			if g[0] != prev[1] || g[1] != prev[2] {
				t.Error("shift error:", string(g), string(prev))
			}
		}

		prev = g
	}
}

func BenchmarkNGram(b *testing.B) {
	var raw [1024 * 10]byte

	for i := 0; i < 1024; i++ {
		copy(raw[i*10:(i+1)*10], []byte("1234567890"))
	}

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		buf := bytes.NewBuffer(raw[:])

		r := NGram{
			r: buf,
			n: 3,
		}

		for {
			g, err := r.Scan()
			if err != nil {
				break
			}

			_ = g
		}
	}
}
