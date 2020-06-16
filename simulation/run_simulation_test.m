function [cho, out, cfcho, cfout, Q] = run_simulation_post_test(model_param, env_param)

    cho = nan(env_param.nsub, env_param.ntrial);
    out = nan(env_param.nsub, env_param.ntrial);
    cfout = nan(env_param.nsub, env_param.ntrial);
    cfcho = nan(env_param.nsub, env_param.ntrial);

    for sub = 1:env_param.nsub

        qlearner = models.QLearningAgent(...
                [model_param.temp(sub), model_param.lr(sub)],...
                model_param.initq,...
                model_param.nstate,...
                model_param.naction,...
                env_param.ntrial,...
                model_param.decision_rule);

        for t = 1:env_param.ntrial

            choice = qlearner.make_choice(env_param.con(t), t);

            outcome = randsample(env_param.rewards,...
                1, true, [
                    1-env_param.p{choice}(env_param.con(t)),...
                    env_param.p{choice}(env_param.con(t))
            ]);

            qlearner.learn(...
                env_param.con(t),...
                choice,...
                outcome...
            );
            
            if model_param.counterfactual
                
                cfchoice = 3 -choice;
                cfoutcome = randsample(env_param.rewards,...
                1, true, [
                    1-env_param.p{cfchoice}(env_param.con(t)),...
                    env_param.p{cfchoice}(env_param.con(t))
                ]);
                
                qlearner.learn(...
                    env_param.con(t),...
                    cfchoice,...
                    cfoutcome...
                );
            
                cfcho(sub, t) = cfchoice;
                cfout(sub, t) = cfoutcome;

            end
            
            out(sub, t) = outcome;
            cho(sub, t) = choice;
            
        end
        
        Q(sub, :, :) = qlearner.Q(:, :);

    end
end