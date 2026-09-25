package router

import (
	"fmt"
	"sort"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
)

type Group struct {
	*baseRouter
}

func NewGroup(conf config.Config, proxylog, upstreamlog *logmon.Monitor) (*Group, error) {
	swapper, err := newGroupSwapper(conf) // FreeToken patch P5: shared with reload
	if err != nil {
		return nil, err
	}

	processes := make(map[string]process.Process, len(swapper.modelToGroup))
	base, err := newBaseRouter("group", conf, processes, proxylog, swapper)
	if err != nil {
		return nil, fmt.Errorf("creating base router: %w", err)
	}
	base.factory = processFactory(upstreamlog, proxylog) // FreeToken patch P5
	base.plannerFor = groupPlannerFor                    // FreeToken patch P5

	for mid := range swapper.modelToGroup {
		modelCfg, _, ok := conf.FindConfig(mid)
		if !ok {
			base.shutdownFn()
			base.procCancel()
			return nil, fmt.Errorf("no model config for %q", mid)
		}
		p, cancel, err := base.newProcess(mid, modelCfg) // FreeToken patch P5: own context
		if err != nil {
			base.shutdownFn()
			base.procCancel()
			return nil, fmt.Errorf("creating process for %q: %w", mid, err)
		}
		processes[mid] = p
		base.procCancels[mid] = cancel
	}

	g := &Group{baseRouter: base}
	go base.run()
	return g, nil
}

// FreeToken patch P5: newGroupSwapper is NewGroup's group bookkeeping, shared
// with reload so a config with a model in two groups is refused either way.
func newGroupSwapper(conf config.Config) (*groupSwapper, error) {
	modelToGroup := make(map[string]string)
	for gid, gcfg := range conf.Routing.Router.Settings.Groups {
		for _, mid := range gcfg.Members {
			if existing, dup := modelToGroup[mid]; dup {
				return nil, fmt.Errorf("model %q is in multiple groups: %q and %q", mid, existing, gid)
			}
			modelToGroup[mid] = gid
		}
	}
	return &groupSwapper{config: conf, modelToGroup: modelToGroup}, nil
}

// FreeToken patch P5: groupPlannerFor is the group router's planner factory
// for reload: the swapper and the models it keeps processes for.
func groupPlannerFor(conf config.Config) (scheduler.Swapper, []string, error) {
	swapper, err := newGroupSwapper(conf)
	if err != nil {
		return nil, nil, err
	}
	members := make([]string, 0, len(swapper.modelToGroup))
	for id := range swapper.modelToGroup {
		members = append(members, id)
	}
	sort.Strings(members)
	return swapper, members, nil
}

// groupSwapper decides evictions from static group configuration.
//
// Same-group siblings are stopped when the group has swap=true. Cross-group
// members are stopped only when the target's group is exclusive; loading a
// model from a non-exclusive group leaves running exclusive groups alone,
// matching the gotcha in the original ProcessGroup behaviour.
type groupSwapper struct {
	config       config.Config
	modelToGroup map[string]string
}

func (p *groupSwapper) EvictionFor(target string, running []string) []string {
	tg := p.modelToGroup[target]
	tgCfg := p.config.Routing.Router.Settings.Groups[tg]

	seen := make(map[string]struct{})
	var result []string
	consider := func(mID string) {
		if mID == target {
			return
		}
		if _, dup := seen[mID]; dup {
			return
		}
		og := p.modelToGroup[mID]
		switch {
		case og == tg && tgCfg.Swap:
			seen[mID] = struct{}{}
			result = append(result, mID)
		// the previous ProcessGroup behaviour did not unload exclusive groups
		// when loading a non-exclusive model. This maintains that gotcha
		// for backwards compatibility. The newer swap matrix approach does not
		// have this issue.
		case og != tg && tgCfg.Exclusive:
			if ogCfg := p.config.Routing.Router.Settings.Groups[og]; !ogCfg.Persistent {
				seen[mID] = struct{}{}
				result = append(result, mID)
			}
		}
	}

	for _, mID := range running {
		consider(mID)
	}
	return result
}

func (p *groupSwapper) OnSwapStart(target string, running []string) {}
