function [parameters, lpp] = runfit(fit_param, method)
    
    options = optimset('display', 'off');
    
    w = waitbar(0, 'Fitting subject');

    tStart = tic;
    for sub = 1:fit_param.nsub

        waitbar(...
            sub/fit_param.nsub,...  % Compute progression
            w,...
            sprintf('%s%d%s%s', 'Fitting subject ', sub, ' with ', method)...
            );

        switch method
            case 'bads'
                [
                    p,...
                    l...
                ] = bads(...
                        @(x) getlpp(...
                        x,...
                        sub,...
                        fit_param),...
                        fit_param.init_value,...
                        fit_param.lb,...
                        fit_param.ub...
                    );

               
            case 'fmincon'
                [
                    p,...
                    l...
                ] = fmincon(...
                        @(x) getlpp(...
                        x,...
                        sub,...
                        fit_param),...
                        fit_param.init_value,...
                        [], [], [], [],...
                        fit_param.lb,...
                        fit_param.ub,...
                        [],...
                        options...
                    );

            otherwise
                disp('Fitting method not recognized');
        end

        parameters(sub, :) = p;
        lpp(sub) = l;

    end
    
    fprintf('%s method time: ', method);
    toc(tStart);
    close(w);
end

function lpp = getlpp(x, isub, fit_param)
    
   qlearner = models.QLearningAgent(...
                [x(1), x(2)],...
                fit_param.initq,...
                fit_param.nstate,...
                fit_param.naction,...
                fit_param.ntrial,...
                fit_param.decision_rule);
    
    nll = qlearner.fit(...
        fit_param.con,...
        fit_param.cho(isub, :),...
        fit_param.cfcho(isub, :),...
        fit_param.out(isub, :),...
        fit_param.cfout(isub, :),...
        fit_param.logLL,...
        fit_param.counterfactual...
     );
    
    p1 = log(gampdf(x(1), fit_param.gam_dist(1), fit_param.gam_dist(2)));
    p2 = log(betapdf(x(2), fit_param.beta_dist(1), fit_param.beta_dist(2)));
    p = -(p1+p2);
    
    lpp = nll + p;
end