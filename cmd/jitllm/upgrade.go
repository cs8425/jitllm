package main

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// upgradeCmd brings each container to this build's version through the
// format's migrations (jlm.Upgrade): in place where a version changed only
// the version word, rewritten from its own pages where it changed tensors.
// The source the container was converted from is not needed.
func upgradeCmd(args []string) error {
	for _, path := range args {
		from, err := jlm.Upgrade(path)
		if err != nil {
			return err
		}
		if from == jlm.Version {
			fmt.Printf("%s is already version %d\n", path, jlm.Version)
			continue
		}
		fmt.Printf("%s upgraded from version %d to %d\n", path, from, jlm.Version)
	}
	return nil
}
