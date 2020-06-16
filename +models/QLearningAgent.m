classdef QLearningAgent < handle
    %QLEARNING Agent
    properties (SetAccess = public)
        Q
        alpha
        beta
        ntrial
        ll
        l
        name
        a
        which_decision_rule
    end
    
    methods
        function obj = QLearningAgent(params, q, nstate, naction, ntrial, ...
                which_decision_rule, name)
            % constructor
            if exist('name', 'var')
                obj.name = name;
            else
                obj.name = 'QLearning';
            end
            obj.Q = ones(nstate, naction) .* q;
            obj.alpha = params(2);
            obj.beta = params(1);
            obj.ntrial = ntrial;
            obj.a = nan(1, ntrial);
            obj.ll = 0;
            obj.which_decision_rule = which_decision_rule;
            
        end
            
        function nll = fit(obj, s, a, cfa, r, cfr, logLL, fit_cf)
            for t = 1:obj.ntrial
                
                p = obj.decision_rule(...
                    s(t)...
                );
               
                if logLL
                    plog = log(p(a(t)));
                else
                    plog = p(a(t));
                end
                
                obj.ll = obj.ll + plog;
                
                
                obj.learn(s(t), a(t), r(t));
                
                if fit_cf
                    obj.learn(s(t), cfa(t), cfr(t));
                end
                              
            end
            
            nll = -obj.ll;
 
        end
        
        function p = decision_rule(obj, s)
            switch (obj.which_decision_rule)
                case 1
                    % softmax
                    p = exp(obj.beta .* obj.Q(s, :)) ...
                    ./sum(exp(obj.beta .*  obj.Q(s, :)));
                
                case 2
                    % argmax
                    if obj.Q(s, 1) ~= obj.Q(s, 2)                        
                        p = double(obj.Q(s, :) == max(obj.Q(s, :)));
                        
                    else
                        p = [0.5, 0.5];
                    end
                otherwise
                    error('not recognized decision rule');
            end
        end               
        
        function choice = make_choice(obj, s, t)
            p = obj.decision_rule(s);
            obj.a(t) = randsample(...
                1:length(p),... % randomly drawn action
                1,... % number of element picked
                true,...% replacement
                p... % probabilities
                );
            
            choice = obj.a(t);
        end
        
        function learn(obj, s, a, r)
            pe = r - obj.Q(s, a);
            
            obj.Q(s, a) = obj.Q(s, a) + obj.alpha * pe;
                                  
        end
        
        
    end
end

